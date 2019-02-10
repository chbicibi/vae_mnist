import numpy as np

import chainer
import chainer.distributions as D
import chainer.functions as F
import chainer.links as L
from chainer import reporter

import common as C_
import net_ae as NA_


################################################################################

class VAELoss(chainer.Chain, NA_.AEMixin):

    def __init__(self, predictor, beta=1.0, k=1, **kwargs):
        super().__init__()
        self.beta = beta
        self.k = k
        self.device = kwargs.get('device', C_.DEVICE)
        n_latent = predictor.n_latent

        with self.init_scope():
            self.predictor = predictor
            self.prior = Prior(n_latent)

        self.n_latent = n_latent
        self.adjust()
        self.name = 'vae'

    def __call__(self, x, x_=None, **kwargs):
        if x_ is None:
            x_ = x
        if self.k > 1:
            x_ = F.repeat(x_, self.k, axis=0)

        q_z = self.encode(x, **kwargs)
        z = self.sample(q_z)
        p_x = self.decode(z, **kwargs)
        p_z = self.prior()

        # 追加誤差関数
        # mse_vel = F.mean_squared_error(x_, p_x.mean)
        # mse_vor = F.mean_squared_error(*map(vorticity, (x_, p_x.mean)))
        # y = F.sigmoid(p_x.mean)
        # mse_vel = self.batch_mean(F.squared_error(*map(logit, (x_, y))))
        # mse_vor = self.batch_mean(F.squared_error(*map(vorticity_logit15,
        #                                                (x_, y))))

        # reporter.report({'mse_vel': mse_vel}, self)
        # reporter.report({'mse_vor': mse_vor}, self)

        # 誤差関数
        reconstr = self.batch_mean(p_x.log_prob(x_))
        # reconstr = -self.batch_mean((p_x.mean - x_) ** 2)
        kl_penalty = self.batch_mean(chainer.kl_divergence(q_z, p_z))
        loss = self.beta * kl_penalty - reconstr

        reporter.report({'loss': loss}, self)
        reporter.report({'reconstr': reconstr}, self)
        reporter.report({'kl_penalty': kl_penalty}, self)

        return loss

    def encode(self, x, **kwargs):
        q_z = self.predictor.encode(x, **kwargs)
        return q_z

    def decode(self, x, **kwargs):
        y = self.predictor.decode(x, **kwargs)
        if kwargs.get('inference'):
            return F.sigmoid(y)

        else:
            p_x = D.Bernoulli(logit=y)
            return p_x

    def sample(self, q_z):
        z = q_z.sample(self.k)
        return F.vstack(z)

    def batch_mean(self, v):
        return F.mean(F.sum(v, axis=tuple(range(1, v.ndim))))

    def predict(self, x, **kwargs):
        x = self.adapt(x)
        with chainer.using_config('train', False), chainer.no_backprop_mode():
            return F.sigmoid(self.predictor(x, inference=True, **kwargs))

    @property
    def link(self):
        return self.predictor


################################################################################

class VAEChain(chainer.Chain, NA_.AEMixin):
    ''' 単層エンコーダ+デコーダ(VAE全結合ネットワーク)
    '''

    def __init__(self, in_size, out_size, activation=F.sigmoid, **kwargs):
        super().__init__()
        self.in_size = in_size
        self.out_size = out_size

        if type(activation) is tuple:
            self.activation_e = None
            self.activation_d = activation[1]
        else:
            self.activation_e = None
            self.activation_d = activation

        self.device = kwargs.get('device', C_.DEVICE)
        self.in_shape = None
        self.init = False
        self.maybe_init(in_size)

    def __call__(self, x, **kwargs):
        h = self.encode(x, **kwargs)
        y = self.decode(h, **kwargs)
        return y

    def encode(self, x, **kwargs):
        # x = self.adapt(x)
        self.in_shape = x.shape[1:]
        self.maybe_init(self.in_shape)

        x_ = x.reshape(-1, self.in_size)
        mu = self.mu(x_)

        if kwargs.get('show_shape'):
            print(f'layer(E{self.name}): in: {x.shape} out: {mu.shape}')

        if kwargs.get('inference'):
            return mu # == D.Normal(loc=mu, log_scale=ln_sigma).mean
        else:
            ln_sigma = self.ln_sigma(x_)  # log(sigma)
            # return mu, ln_sigma
            return D.Normal(loc=mu, log_scale=ln_sigma)

    def decode(self, x, **kwargs):
        # if type(x) is tuple:
        #     x = x[0]
        x = self.adapt(x)
        if isinstance(x, D.Normal):
            x = x.mean
        y = self.dec(x)
        if self.activation_d:
            y = self.activation_d(y)
        y = y.reshape(-1, *self.in_shape)

        if kwargs.get('show_shape'):
            print(f'layer(D{self.name}): in: {x.shape} out: {y.shape}')
        return y

    def maybe_init(self, in_size_):
        if self.init:
            return
        elif in_size_ is None:
            return

        if type(in_size_) is tuple:
            in_size = np.prod(in_size_)
        else:
            in_size = in_size_

        with self.init_scope():
            self.mu = L.Linear(in_size, self.out_size)
            self.ln_sigma = L.Linear(in_size, self.out_size)
            self.dec = L.Linear(self.out_size, in_size)

        self.in_size = in_size
        self.init = True
        self.adjust(device=self.device)


class Prior(chainer.Link):

    def __init__(self, n_latent):
        super().__init__()

        self.loc = np.zeros(n_latent, np.float32)
        self.scale = np.ones(n_latent, np.float32)
        self.register_persistent('loc')
        self.register_persistent('scale')

    def forward(self):
        return D.Normal(self.loc, scale=self.scale)
