# csst2025

## 题目详情
https://nadc.china-vo.org/events/CSSTdatachallenge/info/challenge_7th

## 工具类
https://sites.google.com/cfa.harvard.edu/saoimageds9/download

## 记录录制文件
https://meeting.tencent.com/crm/KejRY6G345 

## 相关资料
Reduction of CCD observations with scanning Fabry-Perot interferometer
https://arxiv.org/abs/astro-ph/0211104

## 思路想法
提供的fits文件就是构建的带有观测噪声的datacube，需要通过这个文件来回答这四个问题：
       （1）请给出[OIII]电离气体外流的空间分布，包括空间尺度、方位角、电离锥张角。
— [OIII]对应的静止波长为5007 Å，需要校准红移，对各像素点的光谱进行发射线的拟合；拟合上了就说明某片区域可能存在[OIII]电离气流。通过谱线的偏离计算中心速度，再计算与中心速度有偏离的地方得到速度分布，以此来识别外流空间。最后分析噪声和误差影响，结合可视化验证。

       （2）请分解出视场范围内气体宽发射线、窄发射线成分的速度场，至少提供Ha和OIII的结果（速度和宽发射线成分的速度弥散）。
       （3）给出OIII发射线气体非参数化运动学特征W80（包括80%流量的发射线轮廓对应线宽）的空间分布。
       （4）请给出利用BPT光谱诊断电离机制的空间分布图。
