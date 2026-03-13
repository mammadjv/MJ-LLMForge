from multiheadattention import MHAttention


class MultiLayerTransformer(nn.Module):
  def __init__(self, num_layers=10, embed_dim=512, num_heads=num_heads, dropout=0.2):
    super(MultiLayerTransformer, self).__init__()

    self.layers = nn.ModuleList()
    for _ in range(num_layers):
      self.layers.append(MHAttention(embed_dim=embed_dim, dropout=dropout, num_heads=num_heads))

  def forward(self, x):
    for layer in self.layers:
      x = layer(x)

    return x

    