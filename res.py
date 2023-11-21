import pickle,os
from tqdm import tqdm
root = 'outputs/xxxx'
lms = []
ss = []
for filename in tqdm(os.listdir(root1)):
    res = pickle.load(open(os.path.join(root1, filename), 'rb'))
    ss.append(res['edit/ss_score'])
    lms.append(res['lms'])
ss = sum(ss) / len(ss)
lms = sum(lms) / len(lms)
# print("gpt2-xl gender: ")
print(f"ss: {ss}, lms: {lms}")
