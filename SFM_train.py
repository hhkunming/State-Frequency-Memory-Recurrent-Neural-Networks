#!/usr/bin/env python

import numpy as np
import cPickle as pickle
from collections import OrderedDict
import sys
import os

lib_path = 'path/to/lib'

sys.path.append(lib_path)
import midi
from midi.utils import midiread, midiwrite

import theano
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

def pre_proc(data):
	norm = np.sqrt(np.sum(data**2, axis=1))
	return data/norm[:, None]

def pre_proc3(data):
	for i in range(data.shape[0]):
		norm = np.sqrt(np.sum(data[i]**2, axis=1))
		data[i] = data[i]/norm[:, None]
	return data

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)

def unzip(zipped):
    new_params = OrderedDict()
    for kk, vv in zipped.items():
        new_params[kk] = vv.get_value()
    return new_params

class MusicDataProvider:
	def __init__(self, dir_path, set_name, pitch_range, dt, batch_size, shuffle=True):
		self.dir_path = dir_path
		self.set_name = set_name
		self.pitch_range = pitch_range
		self.dt = dt
		self.batch_size = batch_size
		self.shuffle = shuffle

		self.curr_epoch = 0
		self.batch_idx = 0

		name_list = os.listdir(self.dir_path+'/'+self.set_name)
		self.data_num = len(name_list)
		self.data = []
		for m in name_list:
			f = self.dir_path+'/'+self.set_name+'/'+m
			wave = midiread(f, self.pitch_range, self.dt)
			self.data.append(wave.piano_roll)

		self.data_list = np.array(range(self.data_num))
		if self.shuffle:
			np.random.shuffle(self.data_list)

		if set_name=='train':
			self.batch_num = self.data_num/self.batch_size
		else:
			self.batch_num = int(np.ceil(1.0*self.data_num/self.batch_size))

	def get_next_batch(self):
		sidx = self.batch_size*self.batch_idx
		data_idx = self.data_list[sidx:sidx+self.batch_size]
		data, nframe = self.make_batch(data_idx)
		curr_epoch = self.curr_epoch
		curr_batch = self.batch_idx
		self.advance_batch()

		return curr_epoch, curr_batch, [data, nframe]

	def advance_batch(self):
		self.batch_idx = (self.batch_idx+1)%self.batch_num
		if self.batch_idx == 0:
			if self.shuffle:
				np.random.shuffle(self.data_list)
			self.curr_epoch += 1

	def make_batch(self, data_idx):
		data = []
		nframe = []
		for idx in data_idx:
			data.append(self.data[idx])
			nframe.append(len(self.data[idx]))
		return np.concatenate(data, axis=0), np.array(nframe)

	def get_batch_num(self):
		return self.batch_num

	def get_data_dims(self):
		return self.data[0].shape[1]


def SFM(tparams, x, omega, opts):
	nsteps = x.shape[0]

	def _recurrence(x_, t_, Re_s_, Im_s_, z_):
		f_ste = T.nnet.sigmoid(T.dot(tparams['W_ste'], z_)+T.dot(tparams['V_ste'], x_)+tparams['b_ste'])
		f_fre = T.nnet.sigmoid(T.dot(tparams['W_fre'], z_)+T.dot(tparams['V_fre'], x_)+tparams['b_fre'])
		f = T.outer(f_ste, f_fre)

		g = T.nnet.sigmoid(T.dot(tparams['W_g'], z_)+T.dot(tparams['V_g'], x_)+tparams['b_g'])
		i = T.tanh(T.dot(tparams['W_i'], z_)+T.dot(tparams['V_i'], x_)+tparams['b_i'])

		Re_s = f*Re_s_+T.outer(g*i, T.cos(omega*t_))
		Im_s = f*Im_s_+T.outer(g*i, T.sin(omega*t_))

		A = T.sqrt(Re_s**2+Im_s**2)

		def __feq(U_o, W_o, V_o, b_o, W_z, b_z, A_k, z_k):
			o = T.nnet.sigmoid(T.dot(U_o, A_k)+T.dot(W_o, z_)+T.dot(V_o, x_)+b_o)
			zz = z_k+o*T.tanh(T.dot(W_z, A_k)+b_z)
			return zz

		res, upd = theano.scan(__feq, sequences=[tparams['U_o'], tparams['W_o'], tparams['V_o'], tparams['b_o'], tparams['W_z'], tparams['b_z'], A.transpose()],
										outputs_info=[T.zeros_like(z_)], name='__feq', n_steps=omega.shape[0])
		return Re_s, Im_s, res[-1]

	rval, updates = theano.scan(_recurrence,
									sequences=[x, (T.arange(nsteps)+1)/nsteps],
									outputs_info=[T.zeros((opts['dim'], opts['dim_feq'])), T.zeros((opts['dim'], opts['dim_feq'])), T.zeros((opts['dim_pitch'],))],
									name='MFO_SFM',
									n_steps=nsteps)
	return rval[2]

def Adaptive_SFM(tparams, x, omega, opts):
	nsteps = x.shape[0]

	def _recurrence(x_, t_, omg_, Re_s_, Im_s_, z_):
		f_ste = T.nnet.sigmoid(T.dot(tparams['W_ste'], z_)+T.dot(tparams['V_ste'], x_)+tparams['b_ste'])
		f_fre = T.nnet.sigmoid(T.dot(tparams['W_fre'], z_)+T.dot(tparams['V_fre'], x_)+tparams['b_fre'])
		f = T.outer(f_ste, f_fre)

		g = T.nnet.sigmoid(T.dot(tparams['W_g'], z_)+T.dot(tparams['V_g'], x_)+tparams['b_g'])
		i = T.tanh(T.dot(tparams['W_i'], z_)+T.dot(tparams['V_i'], x_)+tparams['b_i'])

		omg = T.dot(tparams['W_omg'], z_)+T.dot(tparams['V_omg'], x_)+tparams['b_omg']

		Re_s = f*Re_s_+T.outer(g*i, T.cos(omg_*t_))
		Im_s = f*Im_s_+T.outer(g*i, T.sin(omg_*t_))

		A = T.sqrt(Re_s**2+Im_s**2)

		def __feq(U_o, W_o, V_o, b_o, W_z, b_z, A_k, z_k):
			o = T.nnet.sigmoid(T.dot(U_o, A_k)+T.dot(W_o, z_)+T.dot(V_o, x_)+b_o)
			zz = z_k+o*T.tanh(T.dot(W_z, A_k)+b_z)
			return zz

		res, upd = theano.scan(__feq, sequences=[tparams['U_o'], tparams['W_o'], tparams['V_o'], tparams['b_o'], tparams['W_z'], tparams['b_z'], A.transpose()],
										outputs_info=[T.zeros_like(z_)], name='__feq', n_steps=omega.shape[0])
		return omg, Re_s, Im_s, res[-1]

	rval, updates = theano.scan(_recurrence,
									sequences=[x, (T.arange(nsteps)+1)/nsteps],
									outputs_info=[T.ones(omega.shape)*omega, T.zeros((opts['dim'], opts['dim_feq'])), T.zeros((opts['dim'], opts['dim_feq'])), T.zeros((opts['dim_pitch'],))],
									name='MFO_SFM',
									n_steps=nsteps)
	return rval[3]

def adadelta(tparams, train_feats, train_nframe, omega, cost, grads):

    shared_grads   = [theano.shared(p.get_value() * np.asarray(0.), name='%s_grad' % k)   for k, p in tparams.items()]
    running_grads2 = [theano.shared(p.get_value() * np.asarray(0.), name='%s_rgrad2' % k) for k, p in tparams.items()]
    running_delta2 = [theano.shared(p.get_value() * np.asarray(0.), name='%s_rdelta2' % k)   for k, p in tparams.items()]
    n = len(shared_grads)

    sgup = [(shared_grads[i], grads[i]) for i in range(n)]
    rg2up = [(running_grads2[i], 0.95*running_grads2[i] + 0.05*(grads[i]**2)) for i in range(n)]

    f_grad = theano.function([train_feats, train_nframe, omega], cost, updates=sgup + rg2up,
                                    on_unused_input='ignore', name='adadelta_f_grad', allow_input_downcast=True)

    delta_param = [-T.sqrt(rd2+1e-6) / T.sqrt(rg2+1e-6) * sg  for sg, rd2, rg2 in zip(shared_grads, running_delta2, running_grads2)]
    rd2up = [(rd2, 0.95*rd2 + 0.05*(dp**2)) for rd2, dp in zip(running_delta2, delta_param)]
    paramup = [(p, p+dp) for p, dp in zip(tparams.values(), delta_param)]

    lr = T.scalar('lr')
    f_update = theano.function([lr], [], updates=rd2up + paramup, on_unused_input='ignore', name='adadelta_f_update', allow_input_downcast=True)

    return f_grad, f_update

def eval_batch(batch_fea, nframe, tparams, omega, options):
	x = T.fmatrix('x')
	tp = theano.function([x], SFM(tparams, x, omega, options), name='feat', allow_input_downcast=True)

	n = nframe.shape[0]
	log_like=[None]*n
	for i in range(n):
		x_start = np.sum(nframe[:i])
		x_end = x_start+nframe[i]
		x_input = batch_fea[x_start:x_end]

		outs = tp(x_input)
		log_like_fr = [None]*(nframe[i]-1)
		for j in range(nframe[i]-1):
			log_like_fr[j] = np.dot(x_input[j+1], np.log(softmax(outs[j])))
		log_like[i] = np.mean(log_like_fr)
	return log_like

def build_model_train(tparams, train_opt):
    t_f = T.fmatrix('train_feats')
    t_n = T.ivector('train_nframe')
    omega = T.fvector('omega')

    """ Cost """
    trng = RandomStreams()
    def _cost(i, cost_prev):
        x_start = T.sum(t_n[:i])
        x_end = x_start+t_n[i]
        x = t_f[x_start:x_end]

        outs = SFM(tparams, x, omega, train_opt)
        log_lik = T.nlinalg.trace(T.dot(x[1:], T.log(T.nnet.softmax(outs[:-1])).transpose()))
        
        total_cost = cost_prev+(-log_lik)
        return total_cost

    t_l2 = T.arange(t_n.shape[0], dtype='int64')
    costs, updates = theano.scan(_cost, sequences=[t_l2], outputs_info=T.alloc(np.float64(0.)), name='log_likelyhood', n_steps=t_n.shape[0])
    cost = costs[-1]
    

    """ Gradient """
    grads = theano.grad(cost, wrt=list(tparams.values()))
    

    """ Update Function """
    f_grad, f_update = adadelta(tparams, t_f, t_n, omega, cost, grads)
    return f_grad, f_update

def train(data_opt, train_opt):
	print 'Initializing data provider...'
	dp_tr = MusicDataProvider(data_opt['dir_path'], 'train', data_opt['pitch_range'], data_opt['dt'], data_opt['batch_size'])
	dp_vl = MusicDataProvider(data_opt['dir_path'], 'valid', data_opt['pitch_range'], data_opt['dt'], data_opt['batch_size'], shuffle=False)

	print 'Done. Total training batches:', dp_tr.get_batch_num()

	print 'initializing parameters...'
	input_dim = dp_tr.get_data_dims()
	print 'Input feature dimension:', input_dim

	params = param_init('train', train_opt, input_dim, isAdp=train_opt['isAdp'])
	tparams = OrderedDict()
	for kk, pp in params.items():
		tparams[kk] = theano.shared(params[kk].astype(np.float64), name=kk)

	print 'Building model...'
	f_grad, f_update = build_model_train(tparams, train_opt)
	print [(k, tparams[k].get_value().shape) for k in tparams.keys()]

	print 'Begin training...'
	uidx = 0
	prev_log_like = -np.inf
	omega = np.array([i*2*np.pi/train_opt['dim_feq'] for i in range(train_opt['dim_feq'])])
	for eidx in range(train_opt['max_epoch']):
		for bidx in range(dp_tr.get_batch_num()):
			[epoch, batch, [data, nframe]] = dp_tr.get_next_batch()

			online_cost = f_grad(data, nframe, omega)
			f_update(train_opt['lrate'])
			if bidx%5==0:
				print 'batch {}-{}: {}'.format(eidx, bidx, online_cost)
				sys.stdout.flush()

			if uidx>0 and uidx%train_opt['test_freq']==0:
				print '{} minibatch trained. Begin testing...'.format(uidx)
				test_log = 0
				test_cnt = 0
				for test_bidx in range(dp_vl.get_batch_num()):
					[epoch, batch, [data, nframe]] = dp_vl.get_next_batch()
					log_like = eval_batch(data, nframe, tparams, omega, train_opt)

					test_log += np.sum(log_like)
					test_cnt += nframe.shape[0]
					if test_bidx%10==0:
						print 'batch-{} tested: {}'.format(test_bidx, test_log/test_cnt)
						sys.stdout.flush()
				test_log_like = test_log/test_cnt
				print 'Batch {}-{}, test {} samples, accuracy: {}'.format(eidx, bidx, test_cnt, test_log_like)

				if test_log_like > prev_log_like:
					print 'Best parameter so far found. Saving...'
					param = unzip(tparams)
					fo = open(train_opt['save_dir'], 'wb')
					pickle.dump({'param': param, 'log': test_log_like}, fo, protocol=pickle.HIGHEST_PROTOCOL)
					fo.close()
					prev_log_like = test_log_like
			uidx+=1

def param_init(mode, train_opt, input_dim, isAdp=False):
	if mode == 'train':
		params = OrderedDict()
		params['W_ste'] = pre_proc(np.random.randn(train_opt['dim'], train_opt['dim_pitch']))
		params['V_ste'] = pre_proc(np.random.randn(train_opt['dim'], input_dim))
		params['b_ste'] = np.random.randn(train_opt['dim'])

		params['W_fre'] = pre_proc(np.random.randn(train_opt['dim_feq'], train_opt['dim_pitch']))
		params['V_fre'] = pre_proc(np.random.randn(train_opt['dim_feq'], input_dim))
		params['b_fre'] = np.random.randn(train_opt['dim_feq'])

		params['W_g'] = pre_proc(np.random.randn(train_opt['dim'], train_opt['dim_pitch']))
		params['V_g'] = pre_proc(np.random.randn(train_opt['dim'], input_dim))
		params['b_g'] = np.random.randn(train_opt['dim'])

		params['W_i'] = pre_proc(np.random.randn(train_opt['dim'], train_opt['dim_pitch']))
		params['V_i'] = pre_proc(np.random.randn(train_opt['dim'], input_dim))
		params['b_i'] = np.random.randn(train_opt['dim'])

		params['U_o'] = pre_proc3(np.random.randn(train_opt['dim_feq'], train_opt['dim_pitch'], train_opt['dim']))
		params['W_o'] = pre_proc3(np.random.randn(train_opt['dim_feq'], train_opt['dim_pitch'], train_opt['dim_pitch']))
		params['V_o'] = pre_proc3(np.random.randn(train_opt['dim_feq'], train_opt['dim_pitch'], input_dim))
		params['b_o'] = pre_proc(np.random.randn(train_opt['dim_feq'], train_opt['dim_pitch']))

		params['W_z'] = pre_proc3(np.random.randn(train_opt['dim_feq'], train_opt['dim_pitch'], train_opt['dim']))
		params['b_z'] = pre_proc(np.random.randn(train_opt['dim_feq'], train_opt['dim_pitch']))
		if isAdp:
			params['W_omg'] = pre_proc(np.random.randn(train_opt['dim_feq'], train_opt['dim_pitch']))
			params['V_omg'] = pre_proc(np.random.randn(train_opt['dim_feq'], input_dim))
			params['b_omg'] = np.random.randn(train_opt['dim_feq'])

		return params
	elif mode == 'continue':
		f = pickle.load(file(train_opt['continue_dir']))
		params = f['param']
		return params
  

if __name__ == '__main__':

	data_opt = {'dir_path': '/path/to/dir', 'pitch_range': (21, 109), 'dt': 0.3, 'batch_size': 20}
	train_opt = {'max_epoch': 100, 'lrate': 0.0001, 'dim': 50, 'dim_feq': 4, 'dim_pitch': 88, 'test_freq':15, 'isAdp': False,\
				'continue_dir': '/path/to/continue_dir', 'save_dir': '/path/to/save_dir'}

	train(data_opt, train_opt)

