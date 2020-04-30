# Dependencies
import matplotlib.pyplot as plt # Plotting library
import numpy as np # Algebra library
import tensorflow as tf # Tensorflow
import os # Folder management library
import pandas as pd # Essentially puts data in spreadsheeet
from astropy.io import fits # Reading .fits files
#from PIL import Image # Image processing library
import scipy.misc # General stats functions
import random # Pseudo random number generator
import pickle # Serializing module.
from matplotlib.image import imread # Plotting images
import math



# Functions
def predictions(caps_Lminus1, dims_Lminus1, output_Lminus1, 
                caps_L, dims_L, batch_size, weight_sharing=False, grid_cells_Lminus1=1, name=None):
    
    with tf.name_scope(name, default_name="predictions"):
        if weight_sharing == False:
            output_Lminus1 = tf.squeeze(output_Lminus1, axis=[1, 4], name="output_Lminus1")

        # Initialize first array to random normal
        init_sigma = 0.1
        W_init = tf.random_normal(shape=(1, caps_Lminus1, caps_L, dims_L, dims_Lminus1),
                                  stddev=init_sigma, dtype=tf.float32, name="W_init")
        W = tf.Variable(W_init, name="W")
        tf.summary.histogram("W", W)
        
        # Tile W for the batch
        W_tiled = tf.tile(W, [batch_size, grid_cells_Lminus1, 1, 1, 1], name="W_tiled") 
        # the 1 in the shape argument in tf.tile means keep dimension from W.


        # -1 means add another dimension to the end of tensor
        output_expanded_Lminus1 = tf.expand_dims(output_Lminus1, -1, name="output_expanded_Lminus1")
        # 2 means add another dimension to the 2th position of tensor
        output_tile_Lminus1 = tf.expand_dims(output_expanded_Lminus1, 2, name="output_tile_Lminus1")
        # 1 in shape argument means keep dimensions from caps1_output_tile
        output_tiled_Lminus1 = tf.tile(output_tile_Lminus1, [1, 1, caps_L, 1, 1], name="output_tiled_Lminus1")
        # Get prediction tensor (3rd array) by using t.matmul
        predicted_L = tf.matmul(W_tiled, output_tiled_Lminus1, name="predicted_L")
    
        return predicted_L

    
def routing(caps_Lminus1, caps_L, batch_size, iterations, predicted_L, name=None):
    
    with tf.name_scope(name, default_name="routing"):
        # initialize the raw routing weights to zero
        # The two extra 1s is so that raw_weights and weighted_predictions have the same size
        raw_weights = tf.zeros([batch_size, caps_Lminus1, caps_L, 1, 1], dtype=np.float32, name="raw_weights")

        for r in range(0, iterations):

            # c_i = softmax(b_i)
            routing_weights = tf.nn.softmax(raw_weights, dim=2, name="routing_weights")
            # weighted sum of all the predicted output vectors for each second-layer capsule
            weighted_predictions = tf.multiply(routing_weights, predicted_L, name="weighted_predictions")
            weighted_sum = tf.reduce_sum(weighted_predictions, axis=(1), keepdims=True, name="weighted_sum")

            # Squash weighted_sum (s_j) to get output for 2nd capsules (v_j)
            output_L = squash(weighted_sum, axis=-2, name="output_L")

            # Make caps2_output same size as predicted_caps
            output_tiled_L = tf.tile(output_L, 
                                         [1, caps_Lminus1, 1, 1, 1], 
                                         name="output_tiled_L")
            # the 1 in the shape argument in tf.tile means keep dimension from caps2_output.

            # Dot Product
            agreement = tf.matmul(predicted_L, output_tiled_L, transpose_a=True, name="agreement")
            # Update raw_weights
            raw_weights = tf.add(raw_weights, agreement, name="raw_weights")

        return output_L, routing_weights
        
# safe_norm taken from Aurélien Geron
def squash(s, axis=-1, epsilon=1e-7, name=None):
    
    with tf.name_scope(name, default_name="squash"):
        squared_norm = tf.reduce_sum(tf.square(s), axis=axis,
                                     keepdims=True)
        safe_norm = tf.sqrt(squared_norm + epsilon)
        squash_factor = squared_norm / (1. + squared_norm)
        unit_vector = s / safe_norm
        return squash_factor * unit_vector
    
    
# safe_norm taken from Aurélien Geron
def safe_norm(s, axis=-1, epsilon=1e-7, keep_dims=False, name=None):
    
    with tf.name_scope(name, default_name="safe_norm"):
        squared_norm = tf.reduce_sum(tf.square(s), axis=axis,
                                     keep_dims=keep_dims)
        return tf.sqrt(squared_norm + epsilon)
    
    
    
    
    
    
    
    
# Model

# Placeholder for the input images (batch size, length, width, greyscale).
# Batch size, length, and width are set to None as they can vary. 
X = tf.placeholder(shape=[None, None, None, 1], dtype=tf.float32, name="X")
tf.summary.image('X', X)
y = tf.placeholder(shape=[None], dtype=tf.float32, name="y")


# Hyper-parameters
conv1_params = {
    "filters": 16,
    "kernel_size": 9,
    "strides": 3,
    "padding": "valid",
    "activation": tf.nn.relu,
}
# Convolve image with conv1 filters
conv1 = tf.layers.conv2d(X, name="conv1", **conv1_params)
tf.summary.histogram("conv1", conv1)


# No. of dimenions a single capsule contains.
convcaps_dims = 8 
# No. of capsules per grid cell.
convcaps_caps_types = 32 

convcaps_params = {
    "filters": convcaps_caps_types * convcaps_dims,
    "kernel_size": 5,
    "strides": 2,
    "padding": "valid",
    "activation": tf.nn.relu 
}
# Convolve conv1 with filters.
convcaps = tf.layers.conv2d(conv1, name="convcaps", **convcaps_params)
tf.summary.histogram("convcaps", convcaps)

# Length and width of convaps layer.
convcaps_grid_length = tf.shape(convcaps)[1]   
convcaps_grid_width = tf.shape(convcaps)[2] 
convcaps_grid_cells = convcaps_grid_length * convcaps_grid_width
# Total No. capsules in convcaps layer.
convcaps_caps = convcaps_caps_types * convcaps_grid_cells

# Reshape convcaps so that it is easier to work with.
convcaps_reshape = tf.reshape(convcaps, [-1, convcaps_caps, convcaps_dims], name="convcaps_reshape")
# Squash convcaps with squashing function.
convcaps_output = squash(convcaps_reshape, name="convcaps_output")


# The batch size
batch_size = tf.shape(X)[0]

caps1_caps = 24
caps1_dims = 12

caps1_predictions = predictions(convcaps_caps_types, convcaps_dims, convcaps_output, 
                                caps1_caps, caps1_dims, batch_size, True, convcaps_grid_cells, "caps1_predictions")

caps1_output, routing1 = routing(convcaps_caps, caps1_caps, 
                                  batch_size, 9, caps1_predictions, "routing1")



caps2_caps = 8
caps2_dims = 16

caps2_predictions = predictions(caps1_caps, caps1_dims, caps1_output, 
                                caps2_caps, caps2_dims, batch_size, False, 1, "caps2_predictions")

caps2_output, routing2 = routing(caps1_caps, caps2_caps, batch_size, 9, caps2_predictions, "routing2")



caps3_caps = 1
caps3_dims = 16

caps3_predictions = predictions(caps2_caps, caps2_dims, caps2_output, 
                                caps3_caps, caps3_dims, batch_size, False, 1, "caps3_predictions")


caps3_output, routing3 = routing(caps2_caps, caps3_caps, batch_size, 9, caps3_predictions, "routing3")




with tf.name_scope(name="accuracy_cell"):
    
    # Lengths of capsules represent probability that entity exists.
    lengths = safe_norm(caps3_output, axis=-2, name="lengths")
    # Round lengths to determine predicted class.
    lengths_rounded = tf.round(lengths, name="lengths_rounded")
    # Remove dimensions of size 1 from lengths_rounded
    y_predictions = tf.squeeze(lengths_rounded, axis=[1,2,3], name="y_predictions")
    
    # If y matches y_predictions than 1 otherwise 0.
    correct = tf.equal(y, y_predictions, name="correct")
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32), name="accuracy")
    tf.summary.scalar('accuracy', accuracy)
    
    

m_plus = 0.9
m_minus = 0.1
lambda_ = 0.5


with tf.name_scope(name="loss_cell"):

    T = tf.reshape(y, shape=[-1, 1], name="T")
    # Compute norm of each capsule in digitcaps
    caps3_output_norm = safe_norm(caps3_output, axis=-2, keep_dims=True, name="caps3_output_norm")
    
    present_error_raw = tf.square(tf.maximum(0., m_plus - caps3_output_norm), name="present_error_raw")
    present_error = tf.reshape(present_error_raw, shape=(-1, caps3_caps), name="present_error")
    # -1 tells reshape to calculate the size of this dimension. 

    absent_error_raw = tf.square(tf.maximum(0., caps3_output_norm - m_minus), name="absent_error_raw")
    absent_error = tf.reshape(absent_error_raw, shape=(-1, caps3_caps), name="absent_error")
    # -1 tells reshape to calculate the size of this dimension. 

    # Compute Margin Loss
    L = tf.add(T * present_error, lambda_ * (1.0 - T) * absent_error, name="L")
    loss = tf.reduce_mean(tf.reduce_sum(L, axis=1), name="loss")
    tf.summary.scalar('loss', loss)
    
    

with tf.name_scope(name="train"):
    learning_rate = tf.placeholder(tf.float32, shape=[])
    tf.summary.scalar('learning_rate', learning_rate)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    training_op = optimizer.minimize(loss, name="training_op")
    
merged_summary = tf.summary.merge_all()
init = tf.global_variables_initializer()
saver = tf.train.Saver()



def resize_map(fmap, filter_size, stride, true_image_size):
    fmap_size = fmap.shape[0]
    image_size = (fmap_size - 1)*stride + filter_size
    image = np.zeros((image_size, image_size))
    
    normalizer = np.zeros((image_size, image_size))
    
    for i in range(0, fmap_size):
        for j in range(0, fmap_size):
            y_pos = i*stride
            x_pos = j*stride
            image[0+y_pos:filter_size+y_pos, 0+x_pos:filter_size+x_pos] = image[0+y_pos:filter_size+y_pos, 0+x_pos:filter_size+x_pos] + fmap[i][j]
            
            normalizer[0+y_pos:filter_size+y_pos, 0+x_pos:filter_size+x_pos] = normalizer[0+y_pos:filter_size+y_pos, 0+x_pos:filter_size+x_pos] + 1

    nomarlized_image = image/normalizer
    #padded image
    padded_image = np.hstack((nomarlized_image, np.zeros((nomarlized_image.shape[0], (true_image_size-image_size)))))
    padded_image = np.vstack([padded_image, np.zeros(((true_image_size-image_size), padded_image.shape[1]))])
            
    return padded_image



def visualize_convcaps(caps_and_routing, caps_and_routing_dims, reduce):

    temp_dims = caps_and_routing_dims[0:3] + [1]*(len(caps_and_routing_dims)-3)
    convcaps_lengths = tf.reshape(safe_norm(caps_and_routing[0], axis=-1), temp_dims)
    temp_dims = [1]*(3) + caps_and_routing_dims[3:len(caps_and_routing_dims)]
    convcaps_lengths_tiled = tf.tile(convcaps_lengths, temp_dims)

    temp_dims = caps_and_routing_dims[0:4] + [1]*(len(caps_and_routing_dims)-4)
    routing1_reshape = tf.reshape(caps_and_routing[1], temp_dims)
    temp_dims = [1]*(4) + caps_and_routing_dims[4:len(caps_and_routing_dims)]
    routing1_reshape_tiled = tf.tile(routing1_reshape, temp_dims)

    temp_dims = [1, 1, 1, caps_and_routing_dims[3]] + [1]*(len(caps_and_routing_dims)-4)
    caps1_lengths = tf.reshape(safe_norm(caps_and_routing[2], axis=-2),  temp_dims)
    temp_dims = caps_and_routing_dims[0:3] + [1] + caps_and_routing_dims[4:len(caps_and_routing_dims)]
    caps1_lengths_tiled = tf.tile(caps1_lengths, temp_dims)
    
    temp1 = tf.square(tf.multiply(routing1_reshape_tiled, caps1_lengths_tiled))

    temp_matrices = []
    index = 0
    for i in range(4, len(caps_and_routing_dims)):

        temp_dims = [1]*(i-1) + caps_and_routing_dims[i-1:i+1] + [1]*(len(caps_and_routing_dims)-i-1)
        routing_reshape = tf.reshape(caps_and_routing[3+index], temp_dims)
        temp_dims = caps_and_routing_dims[0:i-1] + [1]*(2) + caps_and_routing_dims[i+1:len(caps_and_routing_dims)]
        routing_reshape_tiled = tf.tile(routing_reshape, temp_dims)

        temp_dims = [1]*len(caps_and_routing_dims)
        temp_dims[i] = caps_and_routing_dims[i]
        caps_lengths = tf.reshape(safe_norm(caps_and_routing[3+index+1], axis=-2), temp_dims)
        temp_dims = caps_and_routing_dims.copy()
        temp_dims[i] = 1
        caps_lengths_tiled = tf.tile(caps_lengths, temp_dims)

        temp = tf.square(tf.multiply(routing_reshape_tiled, caps_lengths_tiled))
        temp_matrices.append(temp)

        index = index + 2
        
    all_paths = tf.multiply(convcaps_lengths_tiled, temp1)
    for temp_matrice in temp_matrices:
        all_paths = tf.multiply(all_paths, temp_matrice)


    normalizing_factor = np.prod(caps_and_routing_dims[2:len(caps_and_routing_dims)])
    all_paths_average = tf.reduce_sum(all_paths, axis=reduce)/normalizing_factor
            
    return all_paths_average



def classify_fits(snle_image_paths, snle_names, start, top_n=45, small_n=10):

    # Get the names of all files in the directory.
#     snle_image_paths = os.listdir(folder_path)
#     snle_names = [x for x in snle_image_paths if x.endswith(".fits")]

#     # Keep only files that end with fits and name them their file path.
#     snle_image_paths = [folder_path+x for x in snle_image_paths if x.endswith(".fits")]
    
    for i, snle_image_path in enumerate(snle_image_paths[start:]):
        # Read fits image data
        fits_list = fits.open(snle_image_path)
        # Make img multiple of 200 in both axis
        fits_image_data = fits_list[0].data[:, :]
        #Add padding so that image is completely divisible by 200x200
        fits_image_data = np.hstack((fits_image_data, np.zeros((fits_image_data.shape[0], 200*math.ceil(fits_image_data.shape[1]/200)-fits_image_data.shape[1]), dtype=fits_image_data.dtype)))
        fits_image_data = np.vstack((fits_image_data, np.zeros((200*math.ceil(fits_image_data.shape[0]/200)-fits_image_data.shape[0], fits_image_data.shape[1]), dtype=fits_image_data.dtype)))
        # Map bad pixels to 0
        fits_image_data[fits_image_data < -30] = 0
        fits_image_data[fits_image_data > 30] = 0
        # Normalize using training data.
        fits_image_data = (fits_image_data - 0.21217042857142857)/10.410399058940191



        crop_size = 200
        top_lengths = []

        all_lengths = []
        length_snle = fits_image_data.shape[0]
        width_snle = fits_image_data.shape[1]

        fmap_size = int(((crop_size-9)/3)+1)
        convcaps_size = int(((fmap_size-5)/2)+1)
        length_ratio = length_snle//crop_size
        width_ratio = width_snle//crop_size

        big_pic = []
        # Crop big image
        cropped_snles = fits_image_data.reshape(length_snle//crop_size,-1,width_snle//crop_size,crop_size).transpose(0,2,1,3)
        cropped_snles = cropped_snles.reshape(-1, crop_size, crop_size)

        batch_size=1
        n_iterations_test = len(cropped_snles) // batch_size
        checkpoint_path = "model/my_capsule_network"

        caps_and_routing = [convcaps_output, 
                            routing1, caps1_output, 
                            routing2, caps2_output, 
                            routing3, caps3_output]
        caps_and_routing_dims = [convcaps_grid_length, convcaps_grid_width, convcaps_caps_types, 
                                 caps1_caps, caps2_caps, caps3_caps]
        reduce = [2, 3, 4]


        with tf.Session() as sess:
            saver.restore(sess, checkpoint_path)

            for iteration in range(1, n_iterations_test + 1):            
                X_batch = cropped_snles[(batch_size*(iteration-1)):(batch_size*iteration)]
                y_batch = np.zeros(batch_size)

                [lengths_] = sess.run([lengths],
                                      feed_dict={X: X_batch.reshape([-1, crop_size, crop_size, 1]),
                                                 y: y_batch}) 


                all_lengths.append(lengths_.flatten()[0])
                # Compute rpv of image if length greater than 0.8
                if lengths_.flatten()[0]>0.80:
                    [routing3_vis] = sess.run([visualize_convcaps(caps_and_routing, caps_and_routing_dims, reduce)], 
                                              feed_dict={X: X_batch.reshape([-1, crop_size, crop_size, 1]),
                                                         y: y_batch})


                    big_pic.append(routing3_vis[:,:,0])

                else:
                    big_pic.append(np.zeros((convcaps_size, convcaps_size)))


        big_pic = np.array(big_pic).reshape(length_ratio, width_ratio, convcaps_size, convcaps_size).transpose(0,2,1,3).reshape(length_ratio*convcaps_size, width_ratio*convcaps_size)
        
        big_pic_count1 = len(np.where( big_pic > 0.00042 )[0]) 
        big_pic_count2 = len(np.where( big_pic > 0.00037 )[0])        
        big_pic_count3 = len(np.where( big_pic > 0.00030 )[0])        
        all_lengths.sort(reverse = True)
        top_length = np.mean(all_lengths[0:top_n])
        small_length = np.mean(all_lengths[0:small_n])
        top_lengths.append(top_length)
        
        print('Count1:%i Count2:%i Count3:%i Avg1:%.2f Avg2:%.2f\n'%(big_pic_count1, big_pic_count2, big_pic_count3, top_length, small_length))

        with open("snle_candidates.txt", "a") as myfile:
            myfile.write(str(snle_names[i+start])+' %i %i %i %f %f\n'%(big_pic_count1, big_pic_count2, big_pic_count3, top_length, small_length))

        plt.figure(figsize=(40, 40))
        plt.imshow(big_pic, vmin=0, vmax=0.00045)
        # plt.colorbar()
        plt.savefig('astro_package_pics/rpv_'+str(snle_names[i+start])+'.png')
        plt.show()
        plt.close()

        print(fits_image_data.shape)
        plt.figure(figsize=(40, 40))
        plt.imshow(fits_image_data, cmap='gray')
        #plt.colorbar()
        plt.savefig('astro_package_pics/snle_'+str(snle_names[i+start])+'.png')
        plt.show()
        plt.close()
        
        with open('start.pickle', 'wb') as f:
            pickle.dump(start+i, f, protocol=pickle.HIGHEST_PROTOCOL)

