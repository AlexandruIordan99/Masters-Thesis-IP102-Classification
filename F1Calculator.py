def F1_Score(precision, recall):
    F1_Score = (2 * precision * recall) / (precision + recall)
    print("F1 Score: ", F1_Score)

F1_Score()


#Because there is no way for Keras to calculate F1-Score, you should use this function and input your
#precision and recall scores