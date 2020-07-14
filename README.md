# ImageProcessing

guideed_filter.py  使用引导滤波（duided filtering）做去雾处理。
video.py  对视频取帧，使用guideed_filter.py处理后，再合并为视频。
save 下面是实验视频，其中1.mp4是原始视频，frames是是取帧（20帧取一）后的图片集，fra是去雾处理后的图片集，test.mp4是处理后新和成的视频。
和save同级的图片是单图去雾处理的测试，2是原图，dark是暗通道图，guide_filter是引导滤波图，t_img是透射率图。
参考博客：
https://www.cnblogs.com/Imageshop/p/3281703.html

https://blog.csdn.net/weixin_43194305/article/details/88959183?utm_medium=distribute.pc_relevant.none-task-blog-BlogCommendFromMachineLearnPai2-2.nonecase&depth_1-utm_source=distribute.pc_relevant.none-task-blog-BlogCommendFromMachineLearnPai2-2.nonecase

https://blog.csdn.net/delw85574/article/details/101491824
