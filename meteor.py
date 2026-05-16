# import sys
# import jieba
# from nltk.translate.meteor_score import meteor_score
#
# def read_file(path):
#     i = 0
#     toks = []
#     with open(path) as f:
#         for line in f.readlines():
#             line = line.strip()
#             toks.append(line)
#             i += 1
#     return toks, i
#
# sys_toks, i1 = read_file(sys.argv[1])
# ref_toks, i2 = read_file(sys.argv[2])
#
# assert i1 == i2, "error"
#
# translations, ref = [], []
# for k in range(i1):
#     translations.append(sys_toks[k])
#     ref.append(ref_toks[k])
#
# meteor_score = meteor_score([translations], ref)
# print(meteor_score)

from nltk.translate.meteor_score import meteor_score
from nltk.tokenize import word_tokenize
import sys
# 文件路径
predictions_file = sys.argv[1]
references_file = sys.argv[2]

# 对整个文本文件进行分词
def tokenize_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f]
    return [word_tokenize(line) for line in lines]

# 分词处理
predictions = tokenize_file(predictions_file)
references = tokenize_file(references_file)
# 读取文件内容
# with open(predictions_file, 'r', encoding='utf-8') as pred_f, \
#      open(references_file, 'r', encoding='utf-8') as ref_f:
#     predictions = [line.strip() for line in pred_f]
#     references = [line.strip() for line in ref_f]

# 确保两个文件的行数一致
assert len(predictions) == len(references), "Prediction and reference files must have the same number of lines."

# 计算每个样本的 METEOR 值
meteor_scores = [meteor_score([ref], pred) for pred, ref in zip(predictions, references)]

# 计算平均 METEOR 值
average_meteor = sum(meteor_scores) / len(meteor_scores)

print(f"Average METEOR Score: {average_meteor:.4f}")
