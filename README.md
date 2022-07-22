<!--
 * @Author: Xuan Wen
 * @Date: 2021-02-22 16:58:15
 * @LastEditTime: 2021-03-30 16:47:02
 * @LastEditors: Please set LastEditors
 * @Description: In User Settings Edit
 * @FilePath: /rnn_sc_wc/README.md
-->
# RNN_SC_WC
---
#### 2021/03/30 update
Representitive Similarity Analysis (for CNN model now)
- Test images
- Train two models
- Generate RSA figures (based on absolute difference between two predicted results)

![Testing image gif](output/test_example.gif)
![RSA](output/RSA_SC-SC_SC-WC_Carte.png)
---
##  Input Image Generator
> Green Dot = Start poke, 0.7   
> Yellow Dot = Target poke, 1   
> Blue Dot = Noise (other poke), 0.2   
> Background, 0   
> Number of noise: 5
#### Self-centered 
![sc3](image/exp_sc3.png)
#### World-centered 
![wc](image/exp_wc1.png)

## CNN training result:
> Same training set and testing set for all conditions    
> 5000 training images, 500 testing images, 5 epoches
#### WC->WC
![](image/input_WC_label_WC_x.png)
![](image/input_WC_label_WC_y.png)
![](image/input_WC_label_WC_mse.png)
![](image/input_WC_label_WC_mae.png)

#### WC->SC
![](image/input_WC_label_SC_x.png)
![](image/input_WC_label_SC_y.png)
![](image/input_WC_label_SC_mse.png)
![](image/input_WC_label_SC_mae.png)

#### SC->WC
![](image/input_SC_label_WC_x.png)
![](image/input_SC_label_WC_y.png)
![](image/input_SC_label_WC_mse.png)
![](image/input_SC_label_WC_mae.png)

#### SC->SC
![](image/input_SC_label_SC_x.png)
![](image/input_SC_label_SC_y.png)
![](image/input_SC_label_SC_mse.png)
![](image/input_SC_label_SC_mae.png)

With the same amount of training, `SC -> WC` gives the worst prediction, which is because it has to learn the distance from `center` to `target poke`, and `center` is not shown on the input image. Following graphs show the `SC->WC` condition with more training epoch.

#### SC->WC with 10 training epoch
![](image/input_SC_label_WC_more__x.png)
![](image/input_SC_label_WC_more__y.png)
![](image/input_SC_label_WC_more__mse.png)
![](image/input_SC_label_WC_more__mae.png)

Training errors are saturated after 8th epoch, and the predictions are still worse than other 3 conditions.

#### SC->WC with 20 training epoch and more activation layer
![](image/input_SC_label_WC_20more_x.png)
![](image/input_SC_label_WC_20more_y.png)
![](image/input_SC_label_WC_20more_mse.png)
![](image/input_SC_label_WC_20more_mae.png)

---

Why does SC -> WC perform much worse than other three conditions?    
**I don't know, computer science is too hard for me**


---
### Improvements:   
Change the representation of pokes from `1x1` to `2x2`:

![](image/input_SC_label_WC_2by2_x.png)
![](image/input_SC_label_WC_2by2_y.png)
![](image/input_SC_label_WC_2by2_mse.png)
![](image/input_SC_label_WC_2by2_mae.png)

it gets much better, probably because information of one pixel gets deluted during the covolution layers? I'm not sure why other three conditions were not affected.

