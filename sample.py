#导入库
import os, torch
import tokenizer
from ckptio import load_ckpt
#设备与权重地址
device = 'cuda' if torch.cuda.is_available() else 'cpu'
HERE = os.path.dirname(os.path.abspath(__file__))
#加载权重（DN 在 FFN 里的下标和 ckpt 不一致时会在 load_ckpt 里自动对齐）
m, meta = load_ckpt(os.path.join(HERE, 'ckpt.pt'), map_location='cpu')
m = m.to(device)
#输出
#提示词
prompt = "SCENE I"
ids = [tokenizer.tokenizerEncoding(prompt)] 
idx = torch.tensor(ids, dtype=torch.long, device=device)
out = m.generate(idx, max_new_tokens=1000)
print(tokenizer.tokenizerDecoding(out[0].tolist()))