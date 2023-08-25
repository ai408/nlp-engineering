def list_metric_test():
    # 第4章/列出可用的评价指标
    from datasets import list_metrics
    metrics_list = list_metrics()
    print(len(metrics_list), metrics_list[:5])


def load_metric_test():
    # 第4章/加载评价指标
    from datasets import load_metric
    metric = load_metric(path="accuracy")  # 加载accuracy指标
    print(metric)

    # 第4章/加载一个评价指标
    from datasets import load_metric
    metric = load_metric(path='glue', config_name='mrpc')  # 加载glue数据集中的mrpc子集
    print(metric)


def load_metric_description_test():
    # 第4章/加载一个评价指标
    from datasets import load_metric
    glue_metric = load_metric('glue', 'mrpc')  # 加载glue数据集中的mrpc子集
    print(glue_metric.inputs_description)

    references = [0, 1]
    predictions = [0, 1]
    results = glue_metric.compute(predictions=predictions, references=references)
    print(results)  # {'accuracy': 1.0, 'f1': 1.0}


if __name__ == "__main__":
    # list_metric_test()

    # load_metric_test()

    load_metric_description_test()