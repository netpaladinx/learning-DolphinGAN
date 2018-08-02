#### modes

* inputs (placeholders):
  * real data `x`: -float32, batch_size(512) x 2
  * past samples `g_old`: -float32, batch_size x 2
    * __ma comes later than pa__
  * noise: `z`: -float32, batch_size x num_z(256)
* `_create_generator` for g
  * 2 hiddens (*128, using relu), out (*2, linear)
* `_create_discriminator` for d_pa, d_ma (seperate)
  * 1 hiddens (*128, using relu), out (*1, softplus)
  * for d_pa
    * real data `x` => d_pa => `d_pa_x`
    * sample `g_new` => d_pa => `d_pa_g_new`
    * objective: pick `x` out of `g_new`
  * for d_ma
    * past sample `g_old` => d_ma => `d_ma_g_old`
    * sample `g_new` => d_ma => `d_ma_g_new`
    * objective: pick `g_new` out of `g_old`
* loss
  * loss for d_pa
    * using GAN
      * `- (log(d_pa_x) + log(1 - d_pa_g_new))`
      * (or) `- (log(d_pa_x) - log(d_pa_g_new))`
    * using D2GAN
      * `- (alpha * log(d_pa_x) - d_pa_g_new)`
  * loss for d_ma
    * using GAN
      * `- (log(d_ma_g_new) + log(1 - d_ma_g_old))`
      * `- (log(d_ma_g_new) - log(d_ma_g_old))`
    * using D2GAN
      * `- (beta * log(d_ma_g_new) - d_ma_g_old)`
  * loss for g
    * using GAN
    * using D2GAN
      * `- (d_pa_g_new + beta * log(d_ma_g_new))`
* optimizer
  * 3 losses: `loss_d_pa`, `loss_d_ma`, `loss_g`
  * parameter collections: `params_d_pa`, `params_d_ma`, `params_g`
  * `learning_rate`(0.0002)
  * Adam (beta1=0.5)
* train
  * inputs:
    * sampler (mix Gaussian) for real data
    * noise signals