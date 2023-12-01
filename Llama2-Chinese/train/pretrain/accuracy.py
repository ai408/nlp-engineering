""" 翻译：精度指标"""
import datasets
from sklearn.metrics import accuracy_score
import evaluate


# 定义精度指标
_DESCRIPTION = """
Accuracy is the proportion of correct predictions among the total number of cases processed. It can be computed with:
Accuracy = (TP + TN) / (TP + TN + FP + FN)
 Where:
TP: True positive
TN: True negative
FP: False positive
FN: False negative
"""

# 定义精度指标的参数
_KWARGS_DESCRIPTION = """
Args:
    predictions (`list` of `int`): Predicted labels.
    references (`list` of `int`): Ground truth labels.
    normalize (`boolean`): If set to False, returns the number of correctly classified samples. Otherwise, returns the fraction of correctly classified samples. Defaults to True.
    sample_weight (`list` of `float`): Sample weights Defaults to None.

Returns:
    accuracy (`float` or `int`): Accuracy score. Minimum possible value is 0. Maximum possible value is 1.0, or the number of examples input, if `normalize` is set to `True`.. A higher score means higher accuracy.

Examples:

    Example 1-A simple example
        >>> accuracy_metric = evaluate.load("accuracy")
        >>> results = accuracy_metric.compute(references=[0, 1, 2, 0, 1, 2], predictions=[0, 1, 1, 2, 1, 0])
        >>> print(results)
        {'accuracy': 0.5}

    Example 2-The same as Example 1, except with `normalize` set to `False`.
        >>> accuracy_metric = evaluate.load("accuracy")
        >>> results = accuracy_metric.compute(references=[0, 1, 2, 0, 1, 2], predictions=[0, 1, 1, 2, 1, 0], normalize=False)
        >>> print(results)
        {'accuracy': 3.0}

    Example 3-The same as Example 1, except with `sample_weight` set.
        >>> accuracy_metric = evaluate.load("accuracy")
        >>> results = accuracy_metric.compute(references=[0, 1, 2, 0, 1, 2], predictions=[0, 1, 1, 2, 1, 0], sample_weight=[0.5, 2, 0.7, 0.5, 9, 0.4])
        >>> print(results)
        {'accuracy': 0.8778625954198473}
"""

# 定义精度指标的引用
_CITATION = """
@article{scikit-learn,
  title={Scikit-learn: Machine Learning in {P}ython},
  author={Pedregosa, F. and Varoquaux, G. and Gramfort, A. and Michel, V.
         and Thirion, B. and Grisel, O. and Blondel, M. and Prettenhofer, P.
         and Weiss, R. and Dubourg, V. and Vanderplas, J. and Passos, A. and
         Cournapeau, D. and Brucher, M. and Perrot, M. and Duchesnay, E.},
  journal={Journal of Machine Learning Research},
  volume={12},
  pages={2825--2830},
  year={2011}
}
"""


# 定义精度指标的类
@evaluate.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION) # 添加文档字符串
class Accuracy(evaluate.Metric):
    def _info(self): # 定义精度指标的信息
        return evaluate.MetricInfo( # 返回精度指标的信息
            description=_DESCRIPTION, # 精度指标的描述
            citation=_CITATION, # 精度指标的引用
            inputs_description=_KWARGS_DESCRIPTION, # 精度指标的输入描述
            features=datasets.Features( # 精度指标的特征
                {
                    "predictions": datasets.Sequence(datasets.Value("int32")), # 预测值
                    "references": datasets.Sequence(datasets.Value("int32")), # 参考值
                }
                if self.config_name == "multilabel" # 如果是多标签
                else {
                    "predictions": datasets.Value("int32"), # 预测值
                    "references": datasets.Value("int32"), # 参考值
                }
            ),
            reference_urls=["https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html"], # 精度指标的参考链接
        )

    def _compute(self, predictions, references, normalize=True, sample_weight=None): # 定义精度指标的计算函数
        return {
            "accuracy": float( # 精度指标
                accuracy_score(references, predictions, normalize=normalize, sample_weight=sample_weight) # 计算精度指标
            )
        }