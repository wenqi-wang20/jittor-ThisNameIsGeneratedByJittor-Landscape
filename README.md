<center>
    <h1>
  	Jittor 草图生成风景比赛 SPADE + FPSE
    </h1>
    <i>ThisNameIsGeneratedByJittor （此名称由计图生成）</i>
</center>

## 队伍成员及分工

* 王文琦：负责算法`FPSE`的实现与仓库维护
* 陈顾骏：负责算法`GauGAN`的实现与报告撰写

## 实现效果

截止到2022年6月29日（A榜封榜前1天），我们采用的算法在 $A$ 榜排名第 $14$，得分 $0.5218$，提交 $request\_id$ 为 $2022062912515137607937$。

我们在训练集上实现的效果如下（左侧为原图像，右侧为生成图像）：

<center>
<img src="https://raw.githubusercontent.com/wenqi-wang20/img/main/img/MDpicturesimage-20220629161446325.png" alt="image-20220629161446325" style="zoom:33%;" />
</center>

我们在测试集上实现的效果如下（左侧为语义标签图，右侧为生成图像）：

<center>
<img src="https://raw.githubusercontent.com/wenqi-wang20/img/main/img/MDpicturesimage-20220629160159114.png" alt="image-20220629160159114" style="zoom: 33%;" />
</center>

## 算法背景

我们主要使用 $Jittor$ 复现了 $GauGAN$（[Semantic~Image~Synthesis~with~Spatially-Adaptive Normalization](https://arxiv.org/abs/1903.07291)）和 $FPSE$（[Learning to Predict Layout-to-image Conditional Convolutions for Semantic Image Synthesis](https://arxiv.org/abs/1910.06809)）的模型结构并成功跑通训练和测试流程，基本复现了原论文的结果。

### $GauGAN$

$GauGAN$，即 $SPADE$ 的主要创新点在于使用了新的 $Spatially-Adaptive~Normalization$ 层来取代传统的 $Batch~Normalization$ 层，以此解决了 $pix2pix$ 等算法中会丢失部分输入语义分割图像信息的问题。主要的修改内容在于 $\gamma$ 和 $\beta$ 的计算不同。

<center>
<img src="https://raw.githubusercontent.com/wenqi-wang20/img/main/img/MDpicturesimg1.png" alt="img1" style="zoom:33%;" />
</center>

在 $Batch~Normalization$ 中 $\gamma$ 和 $\beta$ 的计算是通过网络训练得到的，而 $Spatially-Adaptive~Normalization$ 中 $\gamma$ 和 $\beta$ 是通过语义分割图像计算得到的。

<center>
<img src="https://raw.githubusercontent.com/wenqi-wang20/img/main/img/MDpicturesimg2.png" alt="img2" style="zoom:50%;" />
</center>

<center>
<img src="https://raw.githubusercontent.com/wenqi-wang20/img/main/img/MDpicturesimg3.png" alt="img3" style="zoom:46.5%;" />
</center>

$Spatially-Adaptive~Normalization$ 的极算过程如公式 $(1)$ 所示。在 $Batch~Normalization$ 中， $\gamma$ 和 $\beta$ 是一维张量，其中每个值对应输入特征图的每个通道，而在 $Spatially-Adaptive~Normalization$ 中， $\gamma$ 和 $\beta$ 是三维矩阵，除了通道维度外还有宽和高维度，因此公式 $(1)$ 中 $\gamma$ 和 $\beta$ 下标包含 $c,y,x$ 三个符号。均值μ和标准差σ的计算如公式 $(2)、(3)$ 所示，这部分和 $Batch~Normalization$ 中的计算一样。

<center>
<img src="https://raw.githubusercontent.com/wenqi-wang20/img/main/img/MDpicturesimg4.png" alt="img4" style="zoom: 33%;" />
</center>

网络结构方面，生成器采用堆叠多个 $SPADE ~ResBlk$ 实现，其中每个 $SPADE~ResBlk$ 的结构如左侧所示， $Spatially-Adaptive~Normalization$ 层中的 $\gamma$ 和 $\beta$ 参数通过输入的语义分割图像计算得到。

<center>
<img src="https://raw.githubusercontent.com/wenqi-wang20/img/main/img/MDpicturesimg5.png" alt="img5" style="zoom:50%;" />
</center>

判别器和 $pix2pixHD$ 一样采用常见的 $Patch-GAN$ 形式。

<center>
<img src="https://raw.githubusercontent.com/wenqi-wang20/img/main/img/MDpicturesimg6.png" alt="img6" style="zoom:50%;" />
</center>

从 $SPADE$ 算法的整体示意图来看，生成器的输入可以是一个随机张量，这样生成的图像也是随机的；同样，这个张量也可以通过一个 $Image~Encoder$ 和一张风格图像计算得到，编码网络将输入图像编码成张量，这个张量就包含输入图像的风格，这样就能得到多样化的输出了。

### $FPSE$

$CC-FPSE$ 网络主要是受 $SPACE$ 网络启发而来的。主要使用了一个由权重预测网络预测的条件卷积生成器 G 和一个特征嵌入的鉴别器 D 组成，详细架构如下图所示。

<center>
<img src="https://raw.githubusercontent.com/wenqi-wang20/img/main/img/MDpicturesimage-20220629151511682.png" alt="image-20220629151511682" style="zoom:33%;" />
</center>

在传统的卷积层中，相同的卷积核应用于所有样本和所有空间位置，而不管它们有不同的语义布局。而在 $FPSE$ 网络结构中，认为这种卷及操作对于语义图像的合成不够灵活和有效。所以为了更好地将 semantic image 的布局信息纳入到图像生成的过程中，本篇文章提出了基于语义布局来预测卷积核权值的方法。给定输入特征图 $X \in R^{C \times H \times W}$，通过一个核大小为 $k \times k$ 的卷积层来输出特征图 $Y \in  R ^{D \times H \times W}$。其中使用权值预测网络，使用语义标签作为输入，输出每一个条件卷积层的卷积核权值。**当然，但实际操作的过程中，如果预测所有的卷积核权值，会导致过高的计算成本和 GPU 内存占用**，所以在真实的网络中，只预测轻量级的深度卷积的权值。


<center>
<img src="C:\Users\19749\AppData\Roaming\Typora\typora-user-images\image-20220629153908458.png" alt="image-20220629153908458" style="zoom: 50%;" />
</center>

上图是采取的训练损失函数。其余的鉴别器网络基本与 $SPADE$ 一致。论文中给出的实验效果要优于 $SPADE$。

## 安装

本项目主要运行在单张卡的 3090 上，200个 epoch 的训练周期一般为 4~5 天。

#### 运行环境

- ubuntu 20.04 LTS
- python >= 3.7
- jittor >= 1.3.0

#### 安装依赖

可以进入任意目录执行以下命令安装依赖（ jittor 框架请遵循官网给出的安装指导）

```
pip install -r requirements.txt
```

#### 数据处理

数据目录我们没有上传，请遵循赛事公告进行[下载](https://cloud.tsinghua.edu.cn/f/1d734cbb68b545d6bdf2/?dl=1)。在本次比赛中，我们没有采取更多的数据预处理操作，裁剪、正则化等操作在项目代码中已经有所体现。

预训练模型我们采用的是 `Jittor` 框架自带的 `vgg19` 模型，无需额外下载，在代码运行的过程中会载入到内存里。

## 训练

在单卡上训练，只需执行以下命令（针对 $SPADE$ 和 $FPSE$ 均可）：

```bash
python train.py  \
--name "your project name" \ 
--datasetmode custom \
--label_dir "your train labels directory" \
--image_dir "your train images directory" \
--label_nc 29 \ 
--batchSize "your batchsize" \
--no_instance \
--use_vae
```

因为受平台算力的限制 （单卡3090），$FPSE$ 算法需要更高的参数量，也就需要更大的GPU内存。在实际操作中，$FPSE$ 只能使用 `batchsize = 1` 的梯度下降，导致模型训练效果较佳，但是泛化性能很差；相比之下，$SPADE$ 需要的模型参数量更小，可以使用 `batchsize = 4` 的梯度下降，相应地在测试集上的效果也就更好。我们最终是选择了$SPADE$ 算法的结果上交比赛平台。

## 推断

在单卡上进行测试，只需执行以下命令（针对 $SPADE$ 和 $FPSE$ 均可）：

```bash 
python test.py  \
--name "your project name (the same as the train project)" \ 
--datasetmode custom \
--label_dir "your test labels directory" \
--label_nc 29 \ 
--no_instance \
--use_vae
```

## 致谢

我们将两篇论文的 `pytorch` 版本的源代码，迁移到了 `Jittor` 框架当中。其中借鉴了开源社区 `Spectral Normalization` 的代码，以及重度参考了两篇论文的官方开源代码：[SPADE](https://github.com/NVlabs/SPADE) ，[FPSE](https://github.com/xh-liu/CC-FPSE) 。