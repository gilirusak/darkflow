import os
import time
import numpy as np
import tensorflow as tf
import pickle
from multiprocessing.pool import ThreadPool
import matplotlib.pyplot as plt

train_stats = (
    'Training statistics: \n'
    '\tLearning rate : {}\n'
    '\tBatch size    : {}\n'
    '\tEpoch number  : {}\n'
    '\tBackup every  : {}'
)
pool = ThreadPool()

def _save_ckpt(self, step, loss_profile):
    file = '{}-{}{}'
    model = self.meta['name']

    profile = file.format(model, step, '.profile')
    profile = os.path.join(self.FLAGS.backup, profile)
    with open(profile, 'wb') as profile_ckpt: 
        pickle.dump(loss_profile, profile_ckpt)

    ckpt = file.format(model, step, '')
    ckpt = os.path.join(self.FLAGS.backup, ckpt)
    self.say('Checkpoint at step {}'.format(step))
    self.saver.save(self.sess, ckpt)


def train(self):
    loss_ph = self.framework.placeholders
    loss_mva = None; profile = list()

    batches = self.framework.shuffle()
    loss_op = self.framework.loss

    for i, (x_batch, datum) in enumerate(batches):
        if not i: self.say(train_stats.format(
            self.FLAGS.lr, self.FLAGS.batch,
            self.FLAGS.epoch, self.FLAGS.save
        ))

        feed_dict = {
            loss_ph[key]: datum[key] 
                for key in loss_ph }
        feed_dict[self.inp] = x_batch
        feed_dict.update(self.feed)

        fetches = [self.train_op, loss_op]

        if self.FLAGS.summary:
            fetches.append(self.summary_op)

        fetched = self.sess.run(fetches, feed_dict)
        loss = fetched[1]

        if loss_mva is None: loss_mva = loss
        loss_mva = .9 * loss_mva + .1 * loss
        step_now = self.FLAGS.load + i + 1

        if self.FLAGS.summary:
            self.writer.add_summary(fetched[2], step_now)

        form = 'step {} - loss {} - moving ave loss {}'
        self.say(form.format(step_now, loss, loss_mva))
        profile += [(loss, loss_mva)]

        ckpt = (i+1) % (self.FLAGS.save // self.FLAGS.batch)
        args = [step_now, profile]
        if not ckpt: _save_ckpt(self, *args)

    if ckpt: _save_ckpt(self, *args)

from PIL import Image
import numpy as np

def attack(self):
    # self.framework.loss(self.out)
    print("entering the attack...")
    with self.graph.as_default() as g:
        self.framework.loss(self.out)
    loss_ph = self.framework.placeholders
    loss_mva = None; profile = list()
    print("actual net", self.darknet.layers)
    
    # for layer in self.darknet.layers:
    # if layer.w.dtype != tensorflow.python.ops.init_ops.Constant: 
    # print(layer.w) # print([v.eval(session=self.sess) for v in layer.w.values()])
    zero_out = self.sess.run(self.out, feed_dict={self.inp: np.zeros((1, 416, 416, 3))})
    print(np.sum(np.abs(zero_out)))	
    batches = self.framework.shuffle()
    loss_op = self.framework.loss

    gradients_op = tf.gradients(loss_op, self.inp)[0]
    epsilon = 8.0 / 255.0
    num_steps = 40

    for i, (x_batch, datum) in enumerate(batches):
        adversarial_x = x_batch.copy()
        print("adversarial_x:", np.sum(np.abs(adversarial_x)), adversarial_x.shape)	

        feed_dict = {self.inp: adversarial_x}
        out = self.sess.run(self.out, feed_dict)[0]
        print("out", np.sum(np.abs(out)))
        boxes = self.framework.findboxes(out.copy())
        print("boxes",boxes)

        print(x_batch)
	for k, img in enumerate(x_batch):	
		#print("WRITING TO..." + str(i))
		# adjusted = np.round(x_batch.copy()[1]*255.0) #[:,:,::-1]
		#adjusted = np.round(img * 255.0)
		#img = Image.fromarray(adjusted.astype(np.uint8), 'RGB')
		print("return_predict",self.return_predict(np.asarray(img).astype(np.float32)))
		# img.save('batched_'+str(i*100 + k)+'.png')
		# img.show()
    	for j in range(num_steps):
          feed_dict = {
            loss_ph[key]: datum[key] 
                for key in loss_ph }
          feed_dict[self.inp] = adversarial_x
          # feed_dict.update(self.feed)

          feed_dict = {self.inp: adversarial_x}
      	  gradients_np, loss_np, out_np = self.sess.run([gradients_op, loss_op, self.out], feed_dict=feed_dict)
          out_reshape = out_np.reshape(-1, 13, 13, 5, 6)
          adjusted_c = out_reshape[:, :, :, :, 4]
          adjusted_c = 1.0 / (1.0 + np.exp(-adjusted_c))
          print("max adjusted_c", np.max(adjusted_c))
          print("loss", loss_np)
           # values = sess.run(logits, feed_dict={x_ph:adversarial_x})
    	  # print(values)
          adversarial_x += epsilon / num_steps * np.sign(gradients_np)
          # clipping
       	  adversarial_x = np.clip(np.clip(adversarial_x, x_batch - epsilon, x_batch + epsilon), 0, 1)
	  print("shape", adversarial_x.shape)
	  #adjusted = np.round(adversarial_x[0] * 255.0)
          #img = Image.fromarray(adjusted.astype(np.uint8), 'RGB')
          print("return_predict", self.return_predict(adversarial_x[0]))

          if (j > 0) and ((j % 4) == 0):
              adv_print = np.round(adversarial_x[0] * 255.0)
              pil_img = Image.fromarray(adv_print.astype(np.uint8), 'RGB')
              adversarial_x[0] = np.asarray(pil_img).astype(np.float32) / 255.
              print("return_predict after", self.return_predict(adversarial_x[0]))
	
	# print(adversarial_x[0])
        # plt.plot(adversarial_x[0])
	# print(adversarial_x.shape)
        # return adversarial_x
	# if i == 1:
	for k, img in enumerate(adversarial_x):
		# adv_print = np.round(adversarial_x.copy()[1]*255.0) #[:,:,::-1]
		print("--------------------------------------------------------------")
		#print("Before")
		#print(self.return_predict(img))
		print("shape", img.shape)
		adv_print = np.round(img * 255.0)
		pil_img = Image.fromarray(adv_print.astype(np.uint8), 'RGB')
		print("After")
		img_arr = np.asarray(pil_img).astype(np.float32) / 255.
		print("shape", img_arr.shape)
		print(self.return_predict(img_arr))
		pil_img.save('new_adv_'+str(i*100 + k)+'.png')

def return_predict(self, im):
    assert isinstance(im, np.ndarray), \
				'Image is not a np.ndarray'
    h, w, _ = im.shape
    #im = self.framework.resize_input(im)
    this_inp = np.expand_dims(im, 0)
    feed_dict = {self.inp : this_inp}

    print("img", np.sum(np.abs(this_inp)))

    out = self.sess.run(self.out, feed_dict)[0]
    print("out", np.sum(np.abs(out)))

    out_reshape = out.reshape(-1, 13, 13, 5, 6)
    adjusted_c = out_reshape[:, :, :, :, 4]
    adjusted_c = 1.0 / (1.0 + np.exp(-adjusted_c))
    print("max adjusted_c", np.max(adjusted_c))

    boxes = self.framework.findboxes(out.copy())
    print("boxes",boxes)
    print("confidences", [b.c for b in boxes])
    #print("feed_dict", feed_dict)
    threshold = self.FLAGS.threshold
    boxesInfo = list()
    for box in boxes:
        tmpBox = self.framework.process_box(box, h, w, threshold)
        if tmpBox is None:
            continue
        boxesInfo.append({
            "label": tmpBox[4],
            "confidence": tmpBox[6],
            "topleft": {
                "x": tmpBox[0],
                "y": tmpBox[2]},
            "bottomright": {
                "x": tmpBox[1],
                "y": tmpBox[3]}
        })
    return boxesInfo

import math

def predict(self):
    print("actual net", self.darknet.layers)
    zero_out = self.sess.run(self.out, feed_dict={self.inp: np.zeros((1, 416, 416, 3))})
    print(np.sum(np.abs(zero_out)))

    inp_path = self.FLAGS.imgdir
    all_inps = os.listdir(inp_path)
    all_inps = [i for i in all_inps if self.framework.is_inp(i)]
    if not all_inps:
        msg = 'Failed to find any images in {} .'
        exit('Error: {}'.format(msg.format(inp_path)))

    batch = min(self.FLAGS.batch, len(all_inps))

    # predict in batches
    n_batch = int(math.ceil(len(all_inps) / batch))
    for j in range(n_batch):
        from_idx = j * batch
        to_idx = min(from_idx + batch, len(all_inps))

        # collect images input in the batch
        this_batch = all_inps[from_idx:to_idx]
        inp_feed = pool.map(lambda inp: (
            np.expand_dims(self.framework.preprocess(
                os.path.join(inp_path, inp)), 0)), this_batch)
	#print("inp_feed", inp_feed)
        # Feed to the net
        feed_dict = {self.inp : np.concatenate(inp_feed, 0)}    
        self.say('Forwarding {} inputs ...'.format(len(inp_feed)))

	img = feed_dict[self.inp]
	print("img: {}".format(np.sum(np.abs(img))))

        start = time.time()
        out = self.sess.run(self.out, feed_dict)
        stop = time.time(); last = stop - start
        self.say('Total time = {}s / {} inps = {} ips'.format(
            last, len(inp_feed), len(inp_feed) / last))
	print("out", np.sum(np.abs(out)))

        # Post processing
        self.say('Post processing {} inputs ...'.format(len(inp_feed)))
        start = time.time()
        pool.map(lambda p: (lambda i, prediction:
            self.framework.postprocess(
               prediction, os.path.join(inp_path, this_batch[i])))(*p),
            enumerate(out))
        stop = time.time(); last = stop - start

        # Timing
        self.say('Total time = {}s / {} inps = {} ips'.format(
            last, len(inp_feed), len(inp_feed) / last))
