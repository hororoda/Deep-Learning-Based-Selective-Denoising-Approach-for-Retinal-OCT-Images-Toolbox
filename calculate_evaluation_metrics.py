"""
calculate evaluation metrics
Recall, Specificity, Precision and F1-score for each class
Overall Accuracy, Macro and Micro F1-scores for the model
"""


# target confusion matrix
confusion_matrix_name = ''
# confusion matrix
confusion_matrix = []
# for example :
# confusion_matrix = [[357, 33, 0, 2, 8],
#                     [0, 396, 2, 2, 0],
#                     [0, 6, 389, 0, 5],
#                     [27, 22, 0, 342, 9],
#                     [2, 0, 1, 2, 395]]


# TP, FP, TN, FN initialization
TP = [0, 0, 0, 0, 0]
FP = [0, 0, 0, 0, 0]
TN = [0, 0, 0, 0, 0]
FN = [0, 0, 0, 0, 0]

for i in range(5):
    TP[i] = confusion_matrix[i][i]
    for j in range(5):
        if i ^ j:
            FP[i] += confusion_matrix[j][i]
            for k in range(5):
                if i ^ k:
                    TN[i] += confusion_matrix[j][k]
            FN[i] += confusion_matrix[i][j]

# Micro F1-score
TP_ = sum(TP)
FP_ = sum(FP)
TN_ = sum(TN)
FN_ = sum(FN)
Micro_F1_score = TP_ / (TP_ + 0.5 * (FP_ + FN_))

# RE, SP, PR, F1-score
Recall = [0, 0, 0, 0, 0]
Specificity = [0, 0, 0, 0, 0]
Precision = [0, 0, 0, 0, 0]
F1_score = [0, 0, 0, 0, 0]

for i in range(5):
    Recall[i] = TP[i] / (TP[i] + FN[i])
    Specificity[i] = TN[i] / (FP[i] + TN[i])
    Precision[i] = TP[i] / (TP[i] + FP[i])
    F1_score[i] = 2 * Precision[i] * Recall[i] / (Precision[i] + Recall[i])

# Macro F1-score
Macro_F1_score = sum(F1_score) / len(F1_score)

# Overall Accuracy
Overall_Accuracy = (TP_ + TN_) / (TP_ + FP_ + TN_ + FN_)

# Print
f = open(r'calculate_results.txt', 'a')
print(confusion_matrix_name, file=f)
# classes
diseases = ['AMD', 'CNV', 'DME', 'Drusen', 'Normal']
for i in range(5):
    print('-------%s-------:' % (diseases[i]), file=f)
    print('Recall = %0.4f' % (Recall[i]), file=f)
    print('Specificity = %0.4f' % (Specificity[i]), file=f)
    print('Precision = %0.4f' % (Precision[i]), file=f)
    print('F1-score = %0.4f' % (F1_score[i]), file=f)

print('********************', file=f)
print('Overall Accuracy = %0.4f' % Overall_Accuracy, file=f)
print('Macro F1-score = %0.4f' % Macro_F1_score, file=f)
print('Micro F1-score = %0.4f' % Micro_F1_score, file=f)
print('\n\n', file=f)
f.close()
