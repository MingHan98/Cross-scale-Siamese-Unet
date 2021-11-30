import tensorflow as tf
import labelreg.layers as layer
import labelreg.utils as util


def build_network(network_type, **kwargs):
    type_lower = network_type.lower()
    if type_lower == 'local':
        return LocalNet(**kwargs)
    elif type_lower == 'global':
        return GlobalNet(**kwargs)
    elif type_lower == 'composite':
        return CompositeNet(**kwargs)
    elif type_lower == 'unet':
        return Unet(**kwargs)
    elif type_lower == 'smeunet':
        return SmeUnet(**kwargs)
    elif type_lower == 'resunet':
        return ResUnet(**kwargs)
    elif type_lower == 'multiresunet':
        return MultiResUnet(**kwargs)
    elif type_lower == 'attentionunet':
        return AttentionUnet(**kwargs)
    elif type_lower == 'nolocalunet':
        return NoLocalUnet(**kwargs)
    elif type_lower == 'fcn':
        return FCN(**kwargs)
    elif type_lower == 'vnet':
        return Vnet(**kwargs)
    elif type_lower == 'afunet':
        return AFUnet(**kwargs)


class BaseNet:

    def __init__(self, minibatch_size, image_moving, image_fixed):
        self.minibatch_size = minibatch_size
        self.image_size = image_fixed.shape.as_list()[1:4]
        self.grid_ref = util.get_reference_grid(self.image_size)
        self.grid_warped = tf.zeros_like(self.grid_ref)  # initial zeros are safer for debug
        self.image_moving = image_moving
        self.image_fixed = image_fixed

        self.input_layer = tf.concat([image_moving, image_fixed], axis=4)

    def warp_image(self, input_):
        if input_ is None:
            input_ = self.image_moving
        return util.resample_linear(input_, self.grid_warped)


class SmeBaseNet:

    def __init__(self, minibatch_size, image_moving, image_fixed):
        self.minibatch_size = minibatch_size
        self.image_size = image_moving.shape.as_list()[1:4]
        self.grid_ref = util.get_reference_grid(self.image_size)
        self.grid_warped = tf.zeros_like(self.grid_ref)  # initial zeros are safer for debug
        self.image_moving = image_moving
        self.image_fixed = image_fixed

        self.input_layer_fixed = self.image_fixed
        self.input_layer_moving = self.image_moving

    def warp_image(self, input_):
        if input_ is None:
            input_ = self.image_moving
        return util.resample_linear(input_, self.grid_warped)


class LocalNet(BaseNet):

    def __init__(self, ddf_levels=None, **kwargs):
        BaseNet.__init__(self, **kwargs)
        # defaults
        self.ddf_levels = [0, 1, 2, 3, 4] if ddf_levels is None else ddf_levels
        self.num_channel_initial = 32

        nc = [int(self.num_channel_initial * (2 ** i)) for i in range(5)]
        h0, hc0 = layer.downsample_resnet_block(self.input_layer, 2, nc[0], k_conv0=[7, 7, 7], name='local_down_0')
        h1, hc1 = layer.downsample_resnet_block(h0, nc[0], nc[1], name='local_down_1')
        h2, hc2 = layer.downsample_resnet_block(h1, nc[1], nc[2], name='local_down_2')
        h3, hc3 = layer.downsample_resnet_block(h2, nc[2], nc[3], name='local_down_3')

        hm = [layer.conv3_block(h3, nc[3], nc[4], name='local_deep_4')]

        min_level = min(self.ddf_levels)
        hm += [layer.upsample_resnet_block(hm[0], hc3, nc[4], nc[3], name='local_up_3')] if min_level < 4 else []
        hm += [layer.upsample_resnet_block(hm[1], hc2, nc[3], nc[2], name='local_up_2')] if min_level < 3 else []
        hm += [layer.upsample_resnet_block(hm[2], hc1, nc[2], nc[1], name='local_up_1')] if min_level < 2 else []
        hm += [layer.upsample_resnet_block(hm[3], hc0, nc[1], nc[0], name='local_up_0')] if min_level < 1 else []

        self.ddf = tf.reduce_sum(tf.stack([layer.ddf_summand(hm[4 - idx], nc[idx], self.image_size, name='sum_%d' % idx)
                                           for idx in self.ddf_levels],
                                          axis=5), axis=5)

        self.grid_warped = self.grid_ref + self.ddf


class GlobalNet(BaseNet):

    def __init__(self, **kwargs):
        BaseNet.__init__(self, **kwargs)
        # defaults
        self.num_channel_initial_global = 8
        self.transform_initial = [1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0.]
        nc = [int(self.num_channel_initial_global * (2 ** i)) for i in range(5)]
        h0, hc0 = layer.downsample_resnet_block(self.input_layer, 2, nc[0], k_conv0=[7, 7, 7], name='global_down_0')
        h1, hc1 = layer.downsample_resnet_block(h0, nc[0], nc[1], name='global_down_1')
        h2, hc2 = layer.downsample_resnet_block(h1, nc[1], nc[2], name='global_down_2')
        h3, hc3 = layer.downsample_resnet_block(h2, nc[2], nc[3], name='global_down_3')
        h4 = layer.conv3_block(h3, nc[3], nc[4], name='global_deep_4')

        # h5 = layer.conv3_block(h4, nc[4], nc[4], name='gloabl_deep_5')
        # h6 = layer.conv3_block(h5, nc[4], nc[4], name='gloabl_deep_6')

        theta = layer.fully_connected(h4, 12, self.transform_initial, name='global_project_0')

        self.grid_warped = util.warp_grid(self.grid_ref, theta)
        self.ddf = self.grid_warped - self.grid_ref

        # theta_0 = layer.fully_connected(h4, 12, self.transform_initial, name='global_project_0')
        # theta_1 = layer.fully_connected(h5, 12, self.transform_initial, name='global_project_1')
        # theta_2 = layer.fully_connected(h6, 12, self.transform_initial, name='global_project_2')
        # self.ddf_0 = util.af_warp_grid(self.grid_ref, theta_0)-self.grid_ref
        # self.ddf_1 = util.af_warp_grid(self.grid_ref, theta_1)-self.grid_ref
        # self.ddf_2 = util.af_warp_grid(self.grid_ref, theta_2) - self.grid_ref
        # self.ddf = self.ddf_0*0.2 + self.ddf_1*0.3 + self.ddf_2*0.5
        # self.grid_warped = self.grid_ref + self.ddf


class CompositeNet(BaseNet):

    def __init__(self, **kwargs):
        BaseNet.__init__(self, **kwargs)
        # defaults
        self.ddf_levels = [0]

        global_net = GlobalNet(**kwargs)
        local_net = LocalNet(ddf_levels=self.ddf_levels,
                             minibatch_size=self.minibatch_size,
                             image_moving=global_net.warp_image(self.image_moving),
                             image_fixed=self.image_fixed)
        # local_net = LocalNet(minibatch_size=self.minibatch_size,
        #                     image_moving=global_net.warp_image()
        #                    image_fixed=self.image_fixed,
        #                     ddf_levels=self.ddf_levels)

        self.grid_warped = global_net.grid_warped + local_net.ddf
        self.ddf = self.grid_warped - self.grid_ref


class Unet(BaseNet):
    def __init__(self, **kwargs):
        BaseNet.__init__(self, **kwargs)
        # nc = [32, 64, 128, 256, 512]
        nc = [16, 32, 64, 128, 256]
        h0, hc0 = layer.downsample_Unet_block_1(self.input_layer, 2, nc[0], k_conv0=[7, 7, 7], name='Unet_down_0')
        h1, hc1 = layer.downsample_Unet_block(h0, nc[1], nc[2], name='Unet_down_1')
        h2, hc2 = layer.downsample_Unet_block(h1, nc[2], nc[3], name='Unet_down_2')
        h4 = layer.upsample_Unet_block_1(h2, hc2, nc[3], nc[4], name='Unet_up_0')
        h5 = layer.upsample_Unet_block(h4, hc1, nc[4], nc[3], name='Unet_up_1')
        h6 = layer.upsample_Unet_block(h5, hc0, nc[3], nc[2], name='Unet_up_2')
        h7 = layer.conv3_block(h6, nc[1] + nc[2], nc[1], name='Unet_conv1')
        h8 = layer.conv3_block(h7, nc[1], nc[1], name='Unet_conv2')
        self.ddf = layer.conv3D(h8, nc[1], 3, k_conv=[1, 1, 1], name='DDF')
        self.grid_warped = self.grid_ref + self.ddf


class SmeUnet(SmeBaseNet):
    def __init__(self, **kwargs):
        SmeBaseNet.__init__(self, **kwargs)
        # nc = [32, 64, 128, 256, 512]
        nc = [16, 32, 64, 128, 256]
        f0, fc0 = layer.sme_subnet(self.input_layer_moving, 1, nc[0], k_conv0=[7, 7, 7], name='SmeUnet_fixdown_0')
        f1, fc1 = layer.sme_subnet(f0, nc[0], nc[1], name='SmeUnet_fixdown_1')
        f2, fc2 = layer.sme_subnet(f1, nc[1], nc[2], name='SmeUnet_fixdown_2')
        # f2 = tf.pad(f2, [[0, 0], [0, 2], [0, 0], [0, 1], [0, 0]])
        # f2, fc2 = layer.sme_subnet(f1, nc[1], nc[2], name='SmeUnet_fixdown_2')
        m0, mc0 = layer.sme_subnet(self.input_layer_fixed, 1, nc[0], k_conv0=[7, 7, 7], name='SmeUnet_movdown_0')
        m1, mc1 = layer.sme_subnet(m0, nc[0], nc[1], name='SmeUnet_movdown_1')
        # m2, mc2 = layer.sme_subnet(m1, nc[1], nc[2], use_pooling=0, name='SmeUnet_movdown_2')
        m2, mc2 = layer.sme_subnet(m1, nc[1], nc[2], name='SmeUnet_movdown_2')
        # m2_sup1 = tf.slice(m2, [0, 0, 0, 0, 0], [5, 12, 8, 7, 64])
        # m2_sup2 = tf.slice(m2, [0, 2, 0, 1, 0], [-1, -1, -1, -1, -1])
        # input_centre = tf.concat([f2, m2_sup1, m2_sup2], axis=4)
        input_centre = tf.concat([f2, m2], axis=4)
        h4 = layer.upsample_Unet_block_1(input_centre, fc2, nc[3], nc[3], name='Unet_up_0')
        h5 = layer.upsample_Unet_block(h4, fc1, nc[3], nc[2], name='Unet_up_1')
        h6 = layer.upsample_Unet_block(h5, fc0, nc[2], nc[1], name='Unet_up_2')
        h7 = layer.conv3_block(h6, nc[0] + nc[1], nc[0], name='Unet_conv1')
        h8 = layer.conv3_block(h7, nc[0], nc[0], name='Unet_conv2')
        self.ddf = layer.conv3D(h8, nc[0], 3, k_conv=[1, 1, 1], name='DDF')
        self.grid_warped = self.grid_ref + self.ddf
        # a = tf.compat.v1.pad()


class ResUnet(BaseNet):

    def __init__(self, **kwargs):
        BaseNet.__init__(self, **kwargs)
        # nc = [32, 64, 128, 256, 512]
        nc = [32, 64, 128, 256]
        h0, hc0 = layer.downsample_ResUnet_block_1(self.input_layer, 2, nc[0], k_conv0=[7, 7, 7], name='Unet_down_0')
        h1, hc1 = layer.downsample_ResUnet_block(h0, nc[0], nc[1], name='Unet_down_1')
        h2, hc2 = layer.downsample_ResUnet_block(h1, nc[1], nc[2], name='Unet_down_2')
        h4 = layer.upsample_ResUnet_block_1(h2, hc2, nc[2], nc[3], name='Unet_up_0')
        h5 = layer.upsample_ResUnet_block(h4, hc1, nc[3], nc[2], name='Unet_up_1')
        h6 = layer.upsample_ResUnet_block(h5, hc0, nc[2], nc[1], name='Unet_up_2')
        h7 = layer.conv3_block(h6, nc[1] + nc[0], nc[0], name='Unet_conv1')
        h8 = layer.conv3_block(h7, nc[0], nc[0], name='Unet_conv2')
        self.ddf = layer.conv3D(h8, nc[0], 3, k_conv=[1, 1, 1], name='DDF')
        self.grid_warped = self.grid_ref + self.ddf


class MultiResUnet(BaseNet):

    def __init__(self, **kwargs):
        BaseNet.__init__(self, **kwargs)
        # nc = [32, 64, 128, 256, 512]
        nc = [16, 32, 64, 128]
        self.transform_initial = [1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0.]
        p1, rp1 = layer.downsample_MultiResUnet_block(self.input_layer, 2, nc[0], k_conv0=[7, 7, 7],
                                                      name='MultiResUnet_down_0')
        p2, rp2 = layer.downsample_MultiResUnet_block(p1, nc[0], nc[1], name='MultiResUnet_down_1')
        p3, rp3 = layer.downsample_MultiResUnet_block(p2, nc[1], nc[2], name='MultiResUnet_down_2')
        up1, out_add, h2, h1 = layer.upsample_MultiResUnet_block_1(p3, rp3, nc[2], nc[3], name='MultiResUnet_up_1')
        up2 = layer.upsample_MultiResUnet_block(up1, rp2, nc[3], nc[2], name='MultiResUnet_up_2')
        up3 = layer.upsample_MultiResUnet_block(up2, rp1, nc[2], nc[1], name='MultiResUnet_up_3')
        out = layer.upsample_MultiResUnet_block(up3, rp1, nc[1], nc[0], use_upconv=False, use_concat=False, name='out')
        # h8 = layer.conv3_block(out, nc[0], nc[0], name='Unet_conv2')
        self.ddf = layer.conv3D(out, nc[0], 3, k_conv=[1, 1, 1], name='DDF')

        # theta = tf.sigmoid(layer.affine_layer(hc4, 12, self.transform_initial, name='global_project_0'))
        theta_0 = layer.affine_layer_0(h1, 12, self.transform_initial, name='global_project_0')
        theta_1 = layer.affine_layer_1(h2, 12, self.transform_initial, name='global_project_1')
        theta_2 = layer.affine_layer_2(out_add, 12, self.transform_initial, name='global_project_2')

        # all = tf.constant(3, dtype=tf.float32)
        # # theta = (theta_0*0.2+theta_1*0.3+theta_2*0.5)/all
        # theta = (theta_0 * 0.2 + theta_1 * 0.3 + theta_2 * 0.5) * 0.33333

        self.ddf_0 = util.af_warp_grid(self.grid_ref, theta_0) - self.grid_ref + self.ddf
        self.ddf_1 = util.af_warp_grid(self.grid_ref, theta_1) - self.grid_ref + self.ddf
        self.ddf_2 = util.af_warp_grid(self.grid_ref, theta_2) - self.grid_ref + self.ddf
        self.ddf = self.ddf_0 * 0.2 + self.ddf_1 * 0.3 + self.ddf_2 * 0.5
        self.grid_warped = self.grid_ref + self.ddf


class AttentionUnet(BaseNet):

    def __init__(self, **kwargs):
        BaseNet.__init__(self, **kwargs)
        self.transform_initial = [1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0.]
        # nc = [32, 64, 128, 256, 512]
        nc = [16, 32, 64, 128, 256]
        strides2 = [1, 2, 2, 2, 1]
        h0, hc0 = layer.downsample_Unet_block_1(self.input_layer, 2, nc[0], k_conv0=[7, 7, 7], name='Unet_down_0')
        h1, hc1 = layer.downsample_Unet_block(h0, nc[1], nc[2], name='Unet_down_1')
        h2, hc2 = layer.downsample_Unet_block(h1, nc[2], nc[3], name='Unet_down_2')
        # center = layer.conv3_block(layer.conv3_block(h2, nc[3], nc[3], name='center01'), nc[3], nc[4], name='center02')
        center0 = layer.conv3_block(h2, nc[3], nc[3], name='center01')
        center = layer.conv3_block(center0, nc[3], nc[4], name='center02')
        g1 = layer.UnetGatingSignal(center, name='g1GS')
        attn1 = layer.AttnGatingBlock(hc2, g1, nc[4], name='att1')
        shape1 = h2.shape.as_list()
        up1 = tf.concat([layer.deconv3_block(center, nc[4], nc[4],
                                             shape_out=[shape1[0], shape1[1] * 2, shape1[2] * 2, shape1[3] * 2,
                                                        shape1[4] * 2], strides=strides2, name='u1dc'), attn1], axis=-1)
        center1 = layer.conv3_block(layer.conv3_block(up1, nc[3] + nc[4], nc[3], name='center11'), nc[3], nc[3],
                                    name='center12')

        g2 = layer.UnetGatingSignal(center1, name='g2GS')
        attn2 = layer.AttnGatingBlock(hc1, g2, nc[3], name='att2')
        shape2 = h1.shape.as_list()
        up2 = tf.concat([layer.deconv3_block(center1, nc[3], nc[3],
                                             shape_out=[shape2[0], shape2[1] * 2, shape2[2] * 2, shape2[3] * 2,
                                                        shape2[4] * 2], strides=strides2, name='u2dc'), attn2], axis=-1)
        center2 = layer.conv3_block(layer.conv3_block(up2, nc[3] + nc[2], nc[2], name='center21'), nc[2], nc[2],
                                    name='center22')

        g3 = layer.UnetGatingSignal(center2, name='g3GS')
        attn3 = layer.AttnGatingBlock(hc0, g3, nc[2], name='att3')
        shape3 = h0.shape.as_list()
        up3 = tf.concat([layer.deconv3_block(center2, nc[2], nc[2],
                                             shape_out=[shape3[0], shape3[1] * 2, shape3[2] * 2, shape3[3] * 2,
                                                        shape3[4] * 2], strides=strides2, name='u3dc'), attn3], axis=-1)
        center3 = layer.conv3_block(layer.conv3_block(up3, nc[2] + nc[1], nc[1], name='center31'), nc[1], nc[1],
                                    name='center32')

        self.ddf = layer.conv3D(center3, nc[1], 3, k_conv=[1, 1, 1], name='DDF')

        # theta_0 = layer.affine_layer_0(h2, 12, self.transform_initial, name='global_project_0')
        # theta_1 = layer.affine_layer_1(center0, 12, self.transform_initial, name='global_project_1')
        # theta_2 = layer.affine_layer_2(center, 12, self.transform_initial, name='global_project_2')
        #
        #
        #
        # self.ddf_0 = util.af_warp_grid(self.grid_ref, theta_0)-self.grid_ref+self.ddf
        # self.ddf_1 = util.af_warp_grid(self.grid_ref, theta_1)-self.grid_ref+self.ddf
        # self.ddf_2 = util.af_warp_grid(self.grid_ref, theta_2) - self.grid_ref + self.ddf
        # self.ddf = self.ddf_0*0.2 + self.ddf_1*0.3 + self.ddf_2*0.5
        self.grid_warped = self.grid_ref + self.ddf


class NoLocalUnet(BaseNet):
    def __init__(self, **kwargs):
        BaseNet.__init__(self, **kwargs)
        num_classes = 3
        num_filters_org = 16
        block_sizes = [1] * 3
        block_strides = [1] + [2] * (3 - 1)
        inputs = layer.Conv3D(
            inputs=self.input_layer,
            filters=num_filters_org,
            kernel_size=3,
            strides=1)
        inputs = tf.identity(inputs, 'initial_conv')
        skip_inputs = []
        for i, num_blocks in enumerate(block_sizes):
            # print(i, num_blocks)
            num_filters = num_filters_org * (2 ** i)
            inputs = layer._encoding_block_layer(
                inputs=inputs, filters=num_filters,
                block_fn=layer._residual_block, blocks=num_blocks,
                strides=block_strides[i], training=True,
                name='encode_block_layer{}'.format(i + 1))
            skip_inputs.append(inputs)
        # print(inputs.shape)
        # print(len(skip_inputs))
        inputs = layer.BN_ReLU(inputs, True)
        num_filters = num_filters_org * (2 ** (len(block_sizes) - 1))
        # print(num_filters)
        inputs = layer.multihead_attention_3d(
            inputs, num_filters, num_filters, num_filters, 2, training=True, layer_type='SAME')
        inputs += skip_inputs[-1]
        for i, num_blocks in reversed(list(enumerate(block_sizes[1:]))):
            # print(i, num_blocks)
            num_filters = num_filters_org * (2 ** i)
            if i == len(block_sizes) - 2:
                inputs = layer._att_decoding_block_layer(
                    inputs=inputs, skip_inputs=skip_inputs[i],
                    filters=num_filters, block_fn=layer._residual_block,
                    blocks=1, strides=block_strides[i + 1],
                    training=True,
                    name='decode_block_layer{}'.format(len(block_sizes) - i - 1))
            else:
                inputs = layer._decoding_block_layer(
                    inputs=inputs, skip_inputs=skip_inputs[i],
                    filters=num_filters, block_fn=layer._residual_block,
                    blocks=1, strides=block_strides[i + 1],
                    training=True,
                    name='decode_block_layer{}'.format(len(block_sizes) - i - 1))
        # print(inputs.shape)
        self.ddf = layer._output_block_layer(inputs=inputs, training=True, num_classes=num_classes)
        self.grid_warped = self.grid_ref + self.ddf


class FCN(BaseNet):
    def __init__(self, **kwargs):
        BaseNet.__init__(self, **kwargs)
        nc = [16, 32, 64, 128, 512]
        strides2 = [1, 2, 2, 2, 1]
        strides3 = [1, 8, 8, 8, 1]
        p3, p4, p5 = layer.vgg_net(self.input_layer, nc, name='FCN_PRE')
        con6 = tf.nn.dropout(layer.conv3_block(p5, nc[3], nc[4], k_conv=[7, 7, 7], name='conv6'), keep_prob=0.85)
        con7 = tf.nn.dropout(layer.conv3_block(con6, nc[4], nc[4], k_conv=[1, 1, 1], name='conv7'), keep_prob=0.85)
        con8 = layer.conv3_block(con7, nc[4], 3, k_conv=[1, 1, 1], name='conv8')

        # now to upscale to actual image size
        deconv_shape1 = p4.get_shape()
        b_t1 = layer.bias_variable([deconv_shape1[4].value], name="b_t1")
        conv_deconv1 = layer.fcndeconv3_block(con8, deconv_shape1[4], 3, shape_out=deconv_shape1,
                                              strides=strides2, k_conv=[4, 4, 4], name='FCN_dec1')
        conv_deconv2 = tf.nn.bias_add(conv_deconv1, b_t1)
        fuse_1 = tf.add(conv_deconv2, p4, name='fuse1')

        deconv_shape2 = p3.get_shape()
        b_t2 = layer.bias_variable([deconv_shape2[4].value], name="b_t2")
        conv_deconv2 = layer.fcndeconv3_block(fuse_1, deconv_shape2[4], deconv_shape1[4], shape_out=deconv_shape2,
                                              strides=strides2, k_conv=[4, 4, 4], name='FCN_dec2')
        conv_deconv2 = tf.nn.bias_add(conv_deconv2, b_t2)
        fuse_2 = tf.add(conv_deconv2, p3, name='fuse2')

        # shape = self.input_layer.get_shape()
        # b_t3 = layer.bias_variable([shape[4].value], name="b_t3")
        shape = self.input_layer.shape.as_list()
        shape[4] = 3
        conv_deconv3 = layer.fcndeconv3_block(fuse_2, 3, deconv_shape2[4], shape_out=shape,
                                              strides=strides3, k_conv=[16, 16, 16], name='FCN_dec3')
        self.ddf = layer.conv3D(conv_deconv3, 3, 3, k_conv=[1, 1, 1], name='DDF')
        self.grid_warped = self.grid_ref + self.ddf


class Vnet(BaseNet):
    def __init__(self, **kwargs):
        BaseNet.__init__(self, **kwargs)
        keep_prob = 1.0
        nc = [16, 32, 64, 128, 256]
        num_convolutions = (1, 2, 3, 3)
        bottom_convolutions = 3
        x = self.input_layer
        with tf.variable_scope('vnet/input_layer'):
            x = layer.convolution(x, [7, 7, 7, 2, nc[0]])
            x = tf.layers.batch_normalization(x, momentum=0.99, epsilon=0.001, center=True, scale=True, training=True)
            x = layer.prelu(x)
        features = list()

        for l in range(4):
            with tf.variable_scope('vnet/encoder/level_' + str(l + 1)):
                x = layer.convolution_block(x, num_convolutions[l], keep_prob, activation_fn=layer.prelu,
                                            is_training=True)
                features.append(x)
                with tf.variable_scope('down_convolution'):
                    x = layer.down_convolution(x, factor=2, kernel_size=[2, 2, 2])
                    x = tf.layers.batch_normalization(x, momentum=0.99, epsilon=0.001, center=True, scale=True,
                                                      training=True)
                    x = layer.prelu(x)

        with tf.variable_scope('vnet/bottom_level'):
            x = layer.convolution_block(x, bottom_convolutions, keep_prob, activation_fn=layer.prelu, is_training=True)

        for l in reversed(range(4)):
            with tf.variable_scope('vnet/decoder/level_' + str(l + 1)):
                f = features[l]
                with tf.variable_scope('up_convolution'):
                    x = layer.up_convolution(x, tf.shape(f), factor=2, kernel_size=[2, 2, 2])
                    x = tf.layers.batch_normalization(x, momentum=0.99, epsilon=0.001, center=True, scale=True,
                                                      training=True)
                    x = layer.prelu(x)

                x = layer.convolution_block_2(x, f, num_convolutions[l], keep_prob, activation_fn=layer.prelu,
                                              is_training=True)

        # with tf.variable_scope('vnet/output_layer'):
        #     logits = layer.convolution(x, [1, 1, 1, nc[0], nc[0]])
        #     logits = tf.layers.batch_normalization(logits, momentum=0.99, epsilon=0.001,center=True, scale=True, training=True)
        self.ddf = layer.conv3D(x, nc[0], 3, k_conv=[1, 1, 1], name='DDF')
        self.grid_warped = self.grid_ref + self.ddf


class AFUnet(BaseNet):

    def __init__(self, **kwargs):
        BaseNet.__init__(self, **kwargs)
        # nc = [32, 64, 128, 256, 512]
        self.transform_initial = [1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0.]
        nc = [16, 32, 64, 128, 256]
        h0, hc0 = layer.downsample_Unet_block_1(self.input_layer, 2, nc[0], k_conv0=[7, 7, 7], name='Unet_down_0')
        h1, hc1 = layer.downsample_Unet_block(h0, nc[1], nc[2], name='Unet_down_1')
        h2, hc2 = layer.downsample_Unet_block(h1, nc[2], nc[3], name='Unet_down_2')
        h4, hc4, hc5 = layer.upsample_AFUnet_block_1(h2, hc2, nc[3], nc[4], name='Unet_up_0')
        h5 = layer.upsample_Unet_block(h4, hc1, nc[4], nc[3], name='Unet_up_1')
        h6 = layer.upsample_Unet_block(h5, hc0, nc[3], nc[2], name='Unet_up_2')
        h7 = layer.conv3_block(h6, nc[1] + nc[2], nc[1], name='Unet_conv1')
        h8 = layer.conv3_block(h7, nc[1], nc[1], name='Unet_conv2')
        self.ddf = layer.conv3D(h8, nc[1], 3, k_conv=[1, 1, 1], name='DDF')
        # # theta = tf.sigmoid(layer.affine_layer(hc4, 12, self.transform_initial, name='global_project_0'))
        # theta_0 = layer.affine_layer_0(h2, 12, self.transform_initial, name='global_project_0')
        # theta_1 = layer.affine_layer_1(hc4, 12, self.transform_initial, name='global_project_1')
        # theta_2 = layer.affine_layer_2(hc5, 12, self.transform_initial, name='global_project_2')
        #
        # # all = tf.constant(3, dtype=tf.float32)
        # # # theta = (theta_0*0.2+theta_1*0.3+theta_2*0.5)/all
        # # theta = (theta_0 * 0.2 + theta_1 * 0.3 + theta_2 * 0.5) * 0.33333
        #
        # self.ddf_0 = util.af_warp_grid(self.grid_ref, theta_0)-self.grid_ref+self.ddf
        # self.ddf_1 = util.af_warp_grid(self.grid_ref, theta_1)-self.grid_ref+self.ddf
        # self.ddf_2 = util.af_warp_grid(self.grid_ref, theta_2) - self.grid_ref + self.ddf
        # self.ddf = self.ddf_0*0.2 + self.ddf_1*0.3 + self.ddf_2*0.5
        # self.grid_warped = self.grid_ref + self.ddf
        # 补充实验代码
        # theta = tf.sigmoid(layer.affine_layer(hc4, 12, self.transform_initial, name='global_project_0'))
        theta_0 = layer.affine_layer_0(h2, 12, self.transform_initial, name='global_project_0')
        theta_1 = layer.affine_layer_1(hc4, 12, self.transform_initial, name='global_project_1')
        # theta_2 = layer.affine_layer_2(hc5, 12, self.transform_initial, name='global_project_2')

        # all = tf.constant(3, dtype=tf.float32)
        # # theta = (theta_0*0.2+theta_1*0.3+theta_2*0.5)/all
        # theta = (theta_0 * 0.2 + theta_1 * 0.3 + theta_2 * 0.5) * 0.33333

        self.ddf_0 = util.af_warp_grid(self.grid_ref, theta_0) - self.grid_ref + self.ddf
        self.ddf_1 = util.af_warp_grid(self.grid_ref, theta_1) - self.grid_ref + self.ddf
        # self.ddf_2 = util.af_warp_grid(self.grid_ref, theta_2) - self.grid_ref + self.ddf
        # self.ddf = self.ddf_0*0.2 + self.ddf_1*0.3 + self.ddf_2*0.5
        self.ddf = self.ddf_0 * 0.4 + self.ddf_1 * 0.6
        self.grid_warped = self.grid_ref + self.ddf
