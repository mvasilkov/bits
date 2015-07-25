convnet.js: convnet/*.js
	uglifyjs --compress --mangle --output convnet.js -- \
	    ./convnet/convnet_init.js ./convnet/convnet_util.js \
	    ./convnet/convnet_vol.js ./convnet/convnet_vol_util.js \
	    ./convnet/convnet_layers_dotproducts.js \
	    ./convnet/convnet_layers_pool.js \
	    ./convnet/convnet_layers_input.js \
	    ./convnet/convnet_layers_loss.js \
	    ./convnet/convnet_layers_nonlinearities.js \
	    ./convnet/convnet_layers_dropout.js \
	    ./convnet/convnet_layers_normalization.js \
	    ./convnet/convnet_net.js ./convnet/convnet_trainers.js \
	    ./convnet/convnet_magicnet.js ./convnet/convnet_export.js
