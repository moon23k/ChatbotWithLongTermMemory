def predict_accuracy(sess, cost_op, probability_op, iterator):
    """ Caculate accuracy and loss for dataset
    Args:
        sess: tf.Session
        cost_op: cost operation
        probability_op: probability operation
        iterator: iterator of dataset
    Return:
        accuracy: float32 scalar
        loss: float32 scalar
    """
    n_done = 0
    total_correct = 0
    total_cost = 0
    for instance in iterator:
        n_done += len(instance)
        (batch_x1, batch_x1_mask, batch_x2, batch_x2_mask, batch_y) = prepare_data(instance)

        cost, probability = sess.run([cost_op, probability_op],
                                     feed_dict={"x1:0": batch_x1, "x1_mask:0": batch_x1_mask,
                                                "x2:0": batch_x2, "x2_mask:0": batch_x2_mask,
                                                "y:0": batch_y, "keep_rate:0": 1.0})

        total_correct += (numpy.argmax(probability, axis=1) == batch_y).sum()
        total_cost += cost.sum()

    accuracy = 1.0 * total_correct / n_done
    loss = 1.0 * total_cost / n_done

    return accuracy, loss


def average_precision(sort_data):
    """ calculate average precision (AP)
    If our returned result is 1, 0, 0, 1, 1, 1
    The precision is 1/1, 0, 0, 2/4, 3/5, 4/6
    AP = (1 + 2/4 + 3/5 + 4/6)/4 = 0.69
    Args:
        sort_data: List of tuple, (score, gold_label); score is in [0, 1], glod_label is in {0, 1}
    Return:
        average precision
    """
    count_gold = 0
    sum_precision = 0

    for i, data in enumerate(sort_data):
        if data[1] == 1:
            count_gold += 1
            sum_precision += 1. * count_gold / (i + 1)

    ap = 1. * sum_precision / count_gold

    return ap


def reciprocal_rank(sort_data):
    """ calculate reciprocal rank
    If our returned result is 0, 0, 0, 1, 1, 1
    The rank is 4
    The reciprocal rank is 1/4
    Args:
        sort_data: List of tuple, (score, gold_label); score is in [0, 1], glod_label is in {0, 1}
    Return:
        reciprocal rank
    """

    sort_label = [x[1] for x in sort_data]
    assert 1 in sort_label
    reciprocal_rank = 1. / (1 + sort_label.index(1))

    return reciprocal_rank


def precision_at_position_1(sort_data):
    """ calculate precision at position 1
    Precision= (Relevant_Items_Recommended in top-k) / (k_Items_Recommended)
    Args:
        sort_data: List of tuple, (score, gold_label); score is in [0, 1], glod_label is in {0, 1}
    Return:
        precision_at_position_1
    """

    if sort_data[0][1] == 1:
        return 1
    else:
        return 0


def recall_at_position_k(sort_data, k):
    """ calculate precision at position 1
    Recall= (Relevant_Items_Recommended in top-k) / (Relevant_Items)
    Args:
        sort_data: List of tuple, (score, gold_label); score is in [0, 1], glod_label is in {0, 1}
    Return:
        recall_at_position_k
    """

    sort_label = [s_d[1] for s_d in sort_data]
    gold_label_count = sort_label.count(1)

    select_label = sort_label[:k]
    recall_at_position_k = 1. * select_label.count(1) / gold_label_count

    return recall_at_position_k


def evaluation_one_session(data):
    """ evaluate for one session
    """

    sort_data = sorted(data, key=lambda x: x[0], reverse=True)
    ap = average_precision(sort_data)
    rr = reciprocal_rank(sort_data)
    precision1 = precision_at_position_1(sort_data)
    recall1 = recall_at_position_k(sort_data, 1)
    recall2 = recall_at_position_k(sort_data, 2)
    recall5 = recall_at_position_k(sort_data, 5)

    return ap, rr, precision1, recall1, recall2, recall5


def predict_metrics(sess, cost_op, probability_op, iterator):
    """ Caculate MAP, MRR, Precision@1, Recall@1, Recall@2, Recall@5 
    Args:
        sess: tf.Session
        cost_op: cost operation
        probability_op: probability operation
        iterator: iterator of dataset
    Return:
        metrics: float32 list, [MAP, MRR, Precision@1, Recall@1, Recall@2, Recall@5]
        scores: float32 list, probability for positive label for all instances
    """

    n_done = 0
    scores = []
    labels = []
    for instance in iterator:
        n_done += len(instance)
        (batch_x1, batch_x1_mask, batch_x2, batch_x2_mask, batch_y) = prepare_data(instance)
        cost, probability = sess.run([cost_op, probability_op],
                                     feed_dict={"x1:0": batch_x1, "x1_mask:0": batch_x1_mask,
                                                "x2:0": batch_x2, "x2_mask:0": batch_x2_mask,
                                                "y:0": batch_y, "keep_rate:0": 1.0})

        labels.extend(batch_y.tolist())
        # probability for positive label
        scores.extend(probability[:, 1].tolist())

    assert len(labels) == n_done
    assert len(scores) == n_done

    tf.logging.info("seen samples %s", n_done)

    sum_map = 0
    sum_mrr = 0
    sum_p1 = 0
    sum_r1 = 0
    sum_r2 = 0
    sum_r5 = 0
    total_num = 0

    for i, (s, l) in enumerate(zip(scores, labels)):
        if i % 10 == 0:
            data = []
        data.append((float(s), int(l)))

        if i % 10 == 9:
            total_num += 1
            ap, rr, precision1, recall1, recall2, recall5 = evaluation_one_session(
                data)
            sum_map += ap
            sum_mrr += rr
            sum_p1 += precision1
            sum_r1 += recall1
            sum_r2 += recall2
            sum_r5 += recall5

    metrics = [1. * sum_map / total_num, 1. * sum_mrr / total_num, 1. * sum_p1 / total_num,
               1. * sum_r1 / total_num, 1. * sum_r2 / total_num, 1. * sum_r5 / total_num]

    return metrics, scores