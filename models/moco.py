from models.myres import CifarRes
import torch
import torch.nn as nn
import copy


GLOBAL_QUEUE_SIZE = 65280

class MoCo(nn.Module):
    """
    Build a MoCo model with: a query encoder, a key encoder, and a queue
    https://arxiv.org/abs/1911.05722
    """
    global_queue = nn.functional.normalize(torch.randn(128, GLOBAL_QUEUE_SIZE), dim=0)
    def __init__(self, base_encoder=CifarRes, dim=128, K=65536, m=0.999, T=0.07, mlp=True,
                 use_global_queue=False, client_id=-1):
        """
        dim: feature dimension (default: 128)
        K: queue size; number of negative keys (default: 65536)
        m: moco momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        """
        super(MoCo, self).__init__()

        self.K = K
        self.m = m
        self.T = T
        self.use_global_queue = use_global_queue

        # create the encoders
        # num_classes is the output fc dimension
        self.encoder_q = base_encoder(num_classes=dim)
        self.encoder_k = base_encoder(num_classes=dim)

        if mlp:  # hack: brute-force replacement
            dim_mlp = self.encoder_q.fc.weight.shape[1]
            self.encoder_q.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_q.fc)
            self.encoder_k.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_k.fc)

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        # create the queue
        if not use_global_queue:
            self.register_buffer("queue", torch.randn(dim, K))
            self.queue = nn.functional.normalize(self.queue, dim=0)
        else:
            self.base_ptr = GLOBAL_QUEUE_SIZE // 10 * client_id

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        if use_global_queue:
            self.queue_ptr[0] = self.base_ptr

    def reset_encoder_k(self):
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

    @staticmethod
    def set_global_queue(item):
        MoCo.global_queue = item

    @staticmethod
    def get_global_queue():
        return MoCo.global_queue

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        # gather keys before updating queue
        # keys = concat_all_gather(keys)

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        try:
            if not self.use_global_queue:
                self.queue[:, ptr:ptr + batch_size] = keys.T
            else:
                MoCo.global_queue[:, ptr:ptr + batch_size] = keys.T
        except:
            print("Queue overflow encountered")
            print(keys.shape)
            print(self.queue[:, ptr:ptr + batch_size].shape)
        if not self.use_global_queue:
            ptr = (ptr + batch_size) % self.K  # move pointer
        else:
            ptr = (((ptr - self.base_ptr) + batch_size) % (GLOBAL_QUEUE_SIZE // 10)) + self.base_ptr

        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def _batch_shuffle_ddp(self, x):
        """
        Batch shuffle, for making use of BatchNorm.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        x_gather = x
        batch_size_all = x_gather.shape[0]

        # random shuffle index
        idx_shuffle = torch.randperm(batch_size_all).cuda()

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        return x_gather[idx_shuffle], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x, idx_unshuffle):
        """
        Undo batch shuffle.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        x_gather = x
        batch_size_all = x_gather.shape[0]

        # restored index for this gpu
        idx_this = idx_unshuffle

        return x_gather[idx_this]

    def init_eval(self):
        for param in self.encoder_q.parameters():
            param.requires_grad = False
        self.encoder_qe = copy.deepcopy(self.encoder_q)
        self.encoder_qe.fc = nn.Linear(512, 10)
        self.encoder_qe.eval()

    def forward_eval(self, x):
        return self.encoder_qe(x)

    def forward(self, im_q, im_k):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        Output:
            logits, targets
        """

        # compute query features
        q = self.encoder_q(im_q)  # queries: NxC
        q = nn.functional.normalize(q, dim=1)

        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            # shuffle for making use of BN
            im_k, idx_unshuffle = self._batch_shuffle_ddp(im_k)

            k = self.encoder_k(im_k)  # keys: NxC
            k = nn.functional.normalize(k, dim=1)

            # undo shuffle
            k = self._batch_unshuffle_ddp(k, idx_unshuffle)

        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        # negative logits: NxK
        if not self.use_global_queue:
            l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach().cuda()])
        else:
            l_neg = torch.einsum('nc,ck->nk', [q, MoCo.global_queue.clone().detach().cuda()])

        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)

        # apply temperature
        logits /= self.T

        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

        # dequeue and enqueue
        self._dequeue_and_enqueue(k)

        return logits, labels


# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output



