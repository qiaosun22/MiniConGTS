def get_triplets_set(id_, word_span, tagging_matrix_pred):
    valid_len = word_span[-1][-1] + 1
    st_list = [i[0] for i in word_span]
    end_list = [i[1] for i in word_span]
    triplets_pred = []
    for row in range(valid_len):
        for col in range(valid_len):
            if row == col:
                pass
            elif tagging_matrix_pred[row][col] in args.sentiment2id.values():
                sentiment = int(tagging_matrix_pred[row][col].detach().cpu())
                al, pl = row, col
                ar = al
                pr = pl
                while tagging_matrix_pred[ar+1][pr] == 1:
                    ar += 1
                while tagging_matrix_pred[ar][pr+1] == 1:
                    pr += 1
                    
                if (al in st_list) and (pl in st_list) and (ar in end_list) and (pr in end_list): # ensure the sanity
                    if min(al, ar) > max(pl, pr) or min(pl, pr) > max(al, ar):
                        triplets_pred.append([al, ar, pl, pr, sentiment])

    predicted_set = set()
    for tri in triplets_pred:
        predicted_set.add(str(id_) + '-' + '-'.join(map(str, tri)))
    return predicted_set
