import numpy as np
import keras
from matplotlib import pyplot as plt
import joblib
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

LENGTH_INPUT = 22528

def discriminator(n_input=LENGTH_INPUT):
    model = keras.models.Sequential()
    model.add(keras.layers.Dense(LENGTH_INPUT, input_dim=n_input, activation='relu'))
    model.add(keras.layers.Dense(250, activation='relu', input_dim=n_input))
    model.add(keras.layers.Dense(100, activation='relu'))
    model.add(keras.layers.Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def generator(latent_dim, n_output=LENGTH_INPUT):
    model = keras.models.Sequential()
    model.add(keras.Input(shape=(latent_dim,1)))
    model.add(keras.layers.LSTM(150))
    model.add(keras.layers.Dense(LENGTH_INPUT, activation='linear'))
    model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mean_absolute_error'])
    return model

def gan(generator, discriminator):
    discriminator.trainable = False
    model = keras.models.Sequential()
    model.add(generator)
    model.add(discriminator)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def generate_latent_points(latent_dim, n):
    #generate points
    x_input = np.random.randn(latent_dim*n)
    x_input = x_input.reshape(n, latent_dim)
    return x_input

def generate_fake_samples(generator, latent_dim, n):
    #generate points in latent space
    x_input = generate_latent_points(latent_dim, n)
    #predict outputs
    x = generator.predict(x_input, verbose=0)
    #create class labels
    y = np.zeros((n, 1))
    return x, y

#Train generator and discriminator
def train_gan(g_model, d_model, gan_model, latent_dim, x_real, n_epochs=10000, n_batch=128, n_eval=200):
    #determine half the size of one batch, for updating the discriminator
    half_batch = int(n_batch/2)
    y_real = np.ones((len(x_real), 1))
    for i in range(n_epochs):
        #real samples go here
        x_fake, y_fake = generate_fake_samples(g_model, latent_dim, n=half_batch)
        d_model.train_on_batch(x_real, y_real)
        d_model.train_on_batch(x_fake, y_fake)
        x_gan = generate_latent_points(latent_dim, n=n_batch)
        y_gan = np.ones((n_batch, 1))
        gan_model.train_on_batch(x_gan, y_gan)
        if (i+1) % n_eval == 0:
            """plt.title('Number of epochs: %i'%(i+1))
            pred_data = generate_fake_samples(g_model, latent_dim, n=latent_dim)[0]
            plt.plot(pred_data[0], '.', label='Random fake sample', color='firebrick')
            #plt.plot(x_real[i][0], '.', label='Real fake sample', color='navy')
            plt.legend(fontsize=10)
            plt.show()"""
            print("Don't... don't... don't stop, go on.")

def unpack_labels(y_kk):
    new_y = list()
    for i in y_kk:
        new_y.append(i[0])
    return np.array(new_y)

def get_seizure_by_type(seizure_type, x, y):
    new_x = list()
    for i in range(len(y)):
        if y[i] == seizure_type:
            new_x.append(x[i])
    return np.array(new_x)

"""pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='constant', fill_value=0.0)),
    ('scaler', MinMaxScaler()),
])

x_train, y_train = joblib.load('dwt/train_dwt_four.sav')

x_train_new = pipeline.fit_transform(x_train)
y_train_new = unpack_labels(y_train.to_numpy())

x_mysz = get_seizure_by_type('mysz', x_train_new, y_train_new)
x_cnsz = get_seizure_by_type('cnsz', x_train_new, y_train_new)
x_spsz = get_seizure_by_type('spsz', x_train_new, y_train_new)
x_tnsz = get_seizure_by_type('tnsz', x_train_new, y_train_new)
x_tcsz = get_seizure_by_type('tcsz', x_train_new, y_train_new)
lat_dim = 3
discriminator = discriminator()
generator = generator(latent_dim=lat_dim)
gan_model = gan(generator, discriminator)
train_gan(generator, discriminator, gan_model, lat_dim, x_mysz)

fake_shit = generate_fake_samples(generator, lat_dim, lat_dim)
plt.plot(fake_shit, '.', label='Generated samples', color='firebrick')
plt.plot(x_mysz[0], '.', label='Real samples', color='navy')
plt.show()"""