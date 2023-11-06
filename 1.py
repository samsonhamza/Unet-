from keras import Model
from keras.layers import BatchNormalization, Activation, Add, Multiply, Cropping2D, UpSampling2D, Concatenate
from keras.layers import Conv2D, Conv2DTranspose, MaxPooling2D, concatenate, Input
from keras import backend as K
from keras.layers import Concatenate


def residual_block(inputs, num_filters, kernel_size, strides):
    x = inputs
    x = Conv2D(num_filters, kernel_size, strides=strides, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(num_filters, kernel_size, strides=strides, padding='same')(x)
    x = BatchNormalization()(x)
    x = Add()([x, inputs])
    x = Activation('relu')(x)
    return x
def attention_block(inputs, skip_connection, filters):



    # Define the attention block
    theta_x = Conv2D(filters, 1, activation='relu', padding='same')(inputs)
    theta_x = BatchNormalization()(theta_x)

    phi_g = Conv2D(filters, 1, activation='relu', padding='same')(skip_connection)
    phi_g = BatchNormalization()(phi_g)

    # Upsample theta_x to match the spatial dimensions of phi_g
    theta_x_upsampled = UpSampling2D(size=(1, 1), interpolation='bilinear')(theta_x)
    phi_g_upsampled = UpSampling2D(size=(2, 2))(phi_g)

    concat_xg = Add()([theta_x_upsampled, phi_g_upsampled])
    activation_xg = Activation('relu')(concat_xg)
    psi = Conv2D(1, 1, activation='sigmoid', padding='same')(activation_xg)
    psi = BatchNormalization()(psi)

    # Apply the attention mechanism
    gated = Multiply()([inputs, psi])

    # Upsample the skip connection tensor to match the spatial dimensions
    upsampled_skip_connection = UpSampling2D(size=(2, 2), interpolation='bilinear')(skip_connection)
    return gated, upsampled_skip_connection


def U_net(input_shape, num_filters=64, num_classes=1):
    inputs = Input(input_shape)
    # Encoder
    conv1 = Conv2D(num_filters, 3, activation='relu', padding='same')(inputs)
    conv1 = residual_block(conv1, num_filters, 3, 1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(num_filters * 2, 3, activation='relu', padding='same')(pool1)
    conv2 = residual_block(conv2, num_filters * 2, 3, 1)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(num_filters * 4, 3, activation='relu', padding='same')(pool2)
    conv3 = residual_block(conv3, num_filters * 4, 3, 1)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(num_filters * 8, 3, activation='relu', padding='same')(pool3)
    conv4 = residual_block(conv4, num_filters * 8, 3, 1)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    # Bottleneck
    conv5 = Conv2D(num_filters * 16, 3, activation='relu', padding='same')(pool4)
    conv5 = residual_block(conv5, num_filters * 16, 3, 1)

    # Decoder
    up6 = Conv2DTranspose(num_filters * 8, 2, strides=(2, 2), padding='same', activation='relu')(conv5)
    up6 = Conv2D(num_filters * 8, 3, activation='relu', padding='same')(
        up6)  # add a convolution layer to preserve spatial dimensions
    up6 = Conv2DTranspose(num_filters * 8, 4, strides=(2, 2), padding='same', activation='relu')(
        up6)  # change the upsampling factor to 4
    up6, upsampled_conv4 = attention_block(up6, conv4, num_filters * 8)
    merge6 = Concatenate(axis=3)([upsampled_conv4, up6])
    conv6 = Conv2D(num_filters * 8, 3, activation='relu', padding='same')(merge6)
    conv6 = residual_block(conv6, num_filters * 8, 3, 1)

    up7 = Conv2DTranspose(num_filters * 4, 2, strides=(2, 2), padding='same', activation='relu')(conv6)
    up7, upsampled_conv3 = attention_block(up7, conv3, num_filters * 4)
    merge7 = Concatenate(axis=-1)([upsampled_conv3, up7])
    conv7 = Conv2D(num_filters * 4, 3, activation='relu', padding='same')(merge7)
    conv7 = residual_block(conv7, num_filters * 4, 3, 1)

    up8 = Conv2DTranspose(num_filters * 2, 2, strides=(2, 2), padding='same', activation='relu')(conv7)
    up8, upsampled_conv2 = attention_block(up8, conv2, num_filters * 2)
    merge8 = Concatenate(axis=3)([upsampled_conv2, up8])
    conv8 = Conv2D(num_filters * 2, 3, activation='relu', padding='same')(merge8)
    conv8 = residual_block(conv8, num_filters * 2, 3, 1)

    up9 = Conv2DTranspose(num_filters, 2, strides=(2, 2), padding='same')(conv8)
    up9, upsampled_conv1 = attention_block(up9, conv1, num_filters)
    merge9 = Concatenate(axis=3)([upsampled_conv1, up9])
    conv9 = Conv2D(num_filters, 3, activation='relu', padding='same')(merge9)
    conv9 = residual_block(conv9, num_filters, 3, 1)

    # Output layer
    outputs = Conv2D(num_classes, (1, 1), activation='softmax')(conv9)

    model = Model(inputs=[inputs], outputs=[outputs])
    # Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    return model