import torch
import torch.nn as nn


class LorpConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, r, bias, conv_type, zero_init=False):
        """
        The main convolution is a standard conv2d module specified by `in_channels`, `out_channels`, `kernel_size`.
        The side convolution consists of two types:
            A: 1x1 conv (reduce the `in_channels` to `r`) + 1x1 conv (increase the `r` to `out_channels`)
            B: 1x1 conv (reduce the `in_channels` to `r`) + KxK conv (increase the `r` to `out_channels`)
            , where `K=kernel_size`.
        The inference convolution is initialized as None. When the function `self.re_parameterization()` is called, it
        will be calculated using main convolution and side convolution.
        """
        super(LorpConv2d, self).__init__()
        self.conv_type = conv_type
        self.bias = bias
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.padding = padding

        # Setup main conv.
        self.main_conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=False)

        # Setup side conv.
        self.side_conv = [nn.Conv2d(in_channels, r, kernel_size=1, bias=False)]
        if self.conv_type == 'A':
            self.side_conv.append(nn.Conv2d(r, out_channels, kernel_size=1, bias=bias))
        elif self.conv_type == 'B':
            self.side_conv.append(nn.Conv2d(r, out_channels, kernel_size=kernel_size, padding=padding, bias=bias))
            # Zero initialization according to LoRA.
            if zero_init:
                self.side_conv[-1].weight = nn.Parameter(torch.zeros_like(self.side_conv[-1].weight))
                if bias:
                    self.side_conv[-1].bias = nn.Parameter(torch.zeros_like(self.side_conv[-1].bias))
        self.side_conv = nn.ModuleList(self.side_conv)

        # Setup rep conv.
        self.rep_conv = None
        self._rep_weight = None
        self._rep_bias = None

    def re_parameterization(self):
        conv_m = self.main_conv.weight          # (out, in, K, K)
        conv_s1 = self.side_conv[0].weight      # (r, in, 1, 1)
        conv_s2 = self.side_conv[1].weight      # (out, r, K, K)

        # Merge `conv_s1` and `conv_s2`
        conv_t1 = torch.nn.functional.conv2d(conv_s2, conv_s1.permute(1, 0, 2, 3))
        # Merge `conv_t1`
        pixels_to_pad = (conv_m.shape[-1] - conv_t1.shape[-1]) // 2
        conv_res = nn.functional.pad(conv_t1, [pixels_to_pad, pixels_to_pad, pixels_to_pad, pixels_to_pad])
        conv_res += conv_m

        # Generate `self.rep_conv`
        self._rep_weight = conv_res.clone()
        self._rep_bias = self.side_conv[1].bias
        if self._rep_bias is not None:
            self.rep_conv = lambda x: nn.functional.conv2d(x, self._rep_weight, self._rep_bias, padding=self.padding)
        else:
            self.rep_conv = lambda x: nn.functional.conv2d(x, self._rep_weight, None, padding=self.padding)

    def forward(self, x, mode='train'):
        assert mode in ['train', 'eval']
        if mode == 'train':
            return self.main_conv(x) + self.side_conv[1](self.side_conv[0](x))
        else:
            if self.rep_conv is None:
                raise ValueError('Trying to eval a module before calling `model.re_parameterization()`')
            return self.rep_conv(x)


if __name__ == '__main__':
    in_chan = 3
    out_chan = 4
    ks = 3
    padding_n = 1
    rank = 2
    conv_types = ['A', 'B']
    use_bias = [True, False]
    use_zero_init = [True, False]

    random_input = torch.randn(5, 3, 8, 8).cuda()

    for ct in conv_types:
        for b in use_bias:
            for uz in use_zero_init:
                model = LorpConv2d(in_chan, out_chan, kernel_size=ks, padding=padding_n,
                                   r=rank, conv_type=ct, bias=b, zero_init=uz).cuda()
                output1 = model(random_input, mode='train')
                assert output1.shape == (5, 4, 8, 8)
                model.re_parameterization()
                output2 = model(random_input, mode='eval')
                assert output1.shape == output2.shape
                assert torch.sum((output1 - output2)**2) < 1e-10
