# 1.Problems

### 1.1 Titanic

 - Classification problem: binary classification
 - feature:
   1. continuous variable -> bin





# 2.Experience

- [独热编码的好处](http://blog.csdn.net/wy250229163/article/details/52983760)：

  > 由于分类器往往默认数据数据是连续的，并且是有序的，但是在很多机器学习任务中，存在很多离散（分类）特征，因而将特征值转化成数字时，往往也是不连续的， One-Hot 编码解决了这个问题。 并且，经过独热编码后，特征变成了稀疏的了。这有两个好处，一是解决了分类器不好处理属性数据的问题，二是在一定程度上也起到了扩充特征的作用。

  将离散特征转换为数字时，最好用one-hot编码。

  http://blog.csdn.net/chloezhao/article/details/53484471

  http://pbpython.com/categorical-encoding.html  Approach #3 - One Hot Encoding

  > Label encoding has the advantage that it is straightforward but it has the disadvantage that the numeric values can be “misinterpreted” by the algorithms. For example, the value of 0 is obviously less than the value of 4 but does that really correspond to the data set in real life? Does a wagon have “4X” more weight in our calculation than the convertible? In this example, I don’t think so.
  >
  > A common alternative approach is called one hot encoding (but also goes by several different names shown below).

- [编码和bias项](https://www.cnblogs.com/lianyingteng/p/7792693.html)：

  我们使用one-hot编码时，通常我们的模型不加bias项 或者 加上bias项然后使用L2正则化手段去约束参数；当我们使用哑变量编码时，通常我们的模型都会加bias项，因为不加bias项会导致固有属性的丢失。

- [连续值的离散化为什么会提升模型的非线性能力](https://www.cnblogs.com/lianyingteng/p/7792693.html)

  使用连续值的LR模型用一个权值去管理该特征，而one-hot后有三个权值管理了这个特征，这样使得参数管理的更加精细，所以这样拓展了LR模型的非线性能力。

  这样做除了增强了模型的**非线性能力**外，还有什么好处呢？这样做了我们至少不用再去对变量进行归一化，也可以**加速**参数的更新速度；再者使得一个很大权值管理一个特征，拆分成了许多小的权值管理这个特征多个表示，这样做降低了特征值扰动对模型为**稳定性**影响，也降低了异常数据对模型的影响，进而使得模型具有更好的**鲁棒性**。

- 变量类型 & 处理缺失值

  1. 数值型变量

     连续型变量：用median()填，Age, Fare

     离散型变量：用mode()填，SibSp, Parch.

  2. 类别型变量

     离散型变量：用mode()填， Survived, Sex, and Embarked

  3. 对象型变量

     name

- [Category Encoding](http://pbpython.com/categorical-encoding.html)

  1. Find and Replace

     ```python
     cleanup_nums = {"num_doors":{"four": 4, "two": 2}}
     obj_df.replace(cleanup_nums, inplace=True)
     ```

  2. Label Encoding

     ```python
     obj_df["body_style"] = obj_df["body_style"].astype('category')
     obj_df["body_style_cat"] = obj_df["body_style"].cat.codes
     ```

  3. One Hot Encoding

     ```python
     tar = pd.get_dummies(obj_df, columns=["drive_wheels"])
     ```

  4. Custom Binary Encoding

     ```python
     obj_df["OHC_Code"] = np.where(obj_df["engine_type"].str.contains("ohc"), 1, other=0)
     ```

     ![image](http://pbpython.com/images/np-where-2.png)

  5. Scikit-Learn

     ```python
     from sklearn.preprocessing import LabelEncoder
     lb_make = LabelEncoder()
     obj_df["make_code"] = lb_make.fit_transform(obj_df["make"])

     from sklearn.preprocessing import LabelBinarizer
     lb_style = LabelBinarizer()
     lb_results = lb_style.fit_transform(obj_df["body_style"])
     ```

     ​


# 3.Models

### 3.1 LR

- 优点：
  1. 一是逻辑回归的算法已经比较成熟，预测较为准确；
  2. 二是模型求出的系数易于理解，便于解释，不属于黑盒模型，尤其在银行业，80%的预测使用逻辑回归；
  3. 三是结果是概率值，可以做ranking model；
  4. 四是训练快。
- 缺点：
  1. 分类较多的y都不是很适用；
  2. 对于自变量的多重共线性比较敏感，所以需要利用因子分析或聚类分析来选择代表性的自变量；
  3. 另外预测结果呈现S型，两端概率变化小，中间概率变化大比较敏感，导致很多区间的变量的变化对目标概率的影响没有区分度，无法确定阈值。

