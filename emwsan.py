import itertools
from timm.models.vision_transformer import trunc_normal_
from timm.models.layers import SqueezeExcite
import torch
import torch.nn as nn









def drop_path_f(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output

class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path_f(x, self.drop_prob, self.training)

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x

def _init_vit_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=.01)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out")
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.zeros_(m.bias)
        nn.init.ones_(m.weight)

class Conv(nn.Module):
    def __init__(self, c1, c2, k, s=1):
        super(Conv, self).__init__()
        #aa = int(k // 2)
        self.c = nn.Conv1d(c1, c2, k, s, int(k // 2))
        self.bn = nn.BatchNorm1d(c2)
        self.act = nn.ReLU()

    def forward(self, x):
        return self.act(self.bn(self.c(x)))


def window_partition(x, window_size: int = 16):
    B, L, C = x.shape
    x = x.view(B, L // window_size, window_size, C)
    windows = x.view(-1, window_size, C)
    return windows

def window_reverse(windows, L: int = 192, window_size: int = 16):
    B = int(windows.shape[0] / (L // window_size))
    C = windows.shape[-1]
    x = windows.view(B, -1, window_size, C).view(B, -1, C)
    return x

class PatchEmbed(nn.Module):
    def __init__(self, c1=256, c2=192, k=8, s=8):
        super(PatchEmbed, self).__init__()
        self.conv = nn.Conv1d(c1, c2, k, s)

    def forward(self, x):
        return self.conv(x).transpose(1, 2)

class ConvEmbedEnd(nn.Module):
    def __init__(self, cs=(128, 128, 192), ks=(9, 5, 3)):
        super(ConvEmbedEnd, self).__init__()
        cin = 64
        layers = []
        for k, c in zip(ks, cs):
            layers.append(Conv(cin, c, k))
            cin = c
        self.m = nn.Sequential(*layers)

    def forward(self, x):
        y = self.m(x)
        x = torch.cat([x, y], 1)
        return x

class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        # define a parameter Lambda Table
        self.lanm = nn.Parameter(
            torch.zeros((2 * window_size - 1), num_heads))  # [2*Mh-1, nH]

        lanm_index = []
        for i in range(self.window_size//self.num_heads):
            one_index = torch.arange(i, i + self.window_size).unsqueeze(0)
            i += 1
            lanm_index.append(one_index)
        lanm_index = torch.cat(lanm_index, 0)
        self.register_buffer("lanm_index", lanm_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        qkvs = []
        dws = []
        for i in range(num_heads):
            qkvs.append(nn.Linear(dim, dim))
        self.qkvs = torch.nn.ModuleList(qkvs)

        nn.init.trunc_normal_(self.lanm, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        B_, N, C = x.shape
        feats_in = x.chunk(self.num_heads, dim=1)
        feats_out = []
        feat = feats_in[0]
        for i, qkv in enumerate(self.qkvs):
            if i > 0:  # add the previous output to the input
                feat = feat + feats_in[i]
            q, k, v = (self.qkv(feat).reshape(B_, N//self.num_heads, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)).unbind(0)
            q = q * self.scale
            attn = (q @ k.transpose(-2, -1))
            relative_position_bias = self.lanm[self.lanm_index.view(-1)].view(
                self.window_size//self.num_heads, self.window_size//self.num_heads, -1)
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
            attn = attn + relative_position_bias.unsqueeze(0)[:,0:4, :, :]
            if mask is not None:
                # mask: [nW, ws, ws]
                nW = mask.shape[0]  # num_windows
                aa = mask.unsqueeze(1).unsqueeze(0)

                attn = attn.view(B_ // nW, nW, self.num_heads, N//self.num_heads, N//self.num_heads)+ mask.unsqueeze(1).unsqueeze(0)

                attn = attn.view(-1, self.num_heads, N//self.num_heads,N//self.num_heads)
                attn = self.softmax(attn)
            else:
                attn = self.softmax(attn)

            attn = self.attn_drop(attn)
            feat = (attn @ v).transpose(1, 2).reshape(B_, N//self.num_heads, C)
            feats_out.append(feat)
        x = self.proj(torch.cat(feats_out, 1))
        return x

class SwinTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, window_size=16, shift_size=8,
                 mlp_ratio=1., qkv_bias=True, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)#对最后一个维度进行操作
        self.attn = WindowAttention(dim, window_size=self.window_size, num_heads=num_heads, qkv_bias=qkv_bias,
                                    attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, attn_mask):
        B, L, C = x.shape
        shortcut = x
        x = self.norm1(x)

        # shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size), dims=(1))
        else:
            shifted_x = x
            attn_mask = None

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # [nW*B, L, C]

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=attn_mask)  # [nW*B, L, C]

        # merge windows
        shifted_x = window_reverse(attn_windows, L, self.window_size)  # [B, L, C]

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size), dims=(1))
        else:
            x = shifted_x

        x = shortcut + self.drop_path(x)
        return x

class STBlock(nn.Module):
    def __init__(self, dim=192, depth=2, num_heads=4, window_size=16, token=128//4,
                 mlp_ratio=1., qkv_bias=True, drop=0., attn_drop=0., norm_layer=nn.LayerNorm, ):
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.window_size = window_size
        self.shift_size = window_size // 8
        self.numheads = num_heads
        drop_path = [x.item() for x in torch.linspace(0, 0.05, depth)]

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else self.shift_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer)
            for i in range(depth)])

        self.downsample = None
        aa = next(self.parameters())
        device = next(self.parameters()).device  # get model device
        attn_mask = self.create_mask(token, device)
        self.register_buffer("attn_mask", attn_mask)

    def create_mask(self, L, device):
        img_mask = torch.zeros((1, L, 1), device=device)  # [1, L, 1]
        slices = (slice(0, -self.window_size// self.numheads),
                  slice(-self.window_size // self.numheads , -self.shift_size),
                  slice(-self.shift_size, None))
        cnt = 0
        for s in slices:
            img_mask[:, s, :] = cnt
            cnt += 1
        mask_windows = window_partition(img_mask, self.window_size// self.numheads)  # [nW, window_size, 1]
        mask_windows = mask_windows.view(-1, self.window_size// self.numheads)  # [nW, window_size]
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)  # [nW, 1, window_size] - [nW, window_size, 1]
        # [nW, window_size, window_size]
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        return attn_mask

    def forward(self, x, ):
        for blk in self.blocks:
            x = blk(x, self.attn_mask)
        return x

class EMWSAN(nn.Module):
    def __init__(self, num_classes=7,act_layer=nn.GELU):
        super(EMWSAN, self).__init__()
        self.conv1 = Conv(1, 64, 15, 2)
        self.conv_embend = ConvEmbedEnd()
        self.patch_emb = PatchEmbed()
        self.convf = nn.Conv1d(1, 1, 15, 1, 132)
        self.stblock = STBlock()
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.norm = nn.LayerNorm(192)
        self.head1 = nn.Linear(192, 64)
        self.drop = nn.Dropout(0.2)
        self.head2 = nn.Linear(64, num_classes)
        self.head3 = nn.Linear(1798, 2048)
        self.norm2 = nn.LayerNorm(192)
        self.mlp = Mlp(in_features=192, hidden_features=264, act_layer=act_layer)

        self.apply(self._init_weights)

    def _init_weights(self, m):#逐层的去初始化每层的参数
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out")
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x, vis=False):
        #with torch.no_grad():
        feas = []
        x = x.unsqueeze(1)
        x = self.convf(x)
        x = self.conv1(x)
        x = self.conv_embend(x)
        x = self.patch_emb(x)
        x = self.stblock(x)
        #FFN
        x = x + self.mlp(self.norm2(x))
        x = self.norm(x)
        x = self.avgpool(x.transpose(1, 2))
        x = torch.flatten(x, 1)
        x = self.drop(self.head1(x))
        x = self.head2(x)
        feas.append(x)
        return x if not vis else [x, feas]
