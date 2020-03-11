#' Load Pre-trained Imagenet Model
#'
#' @param ... Other parameters to `keras::application_inception_v3`
#'
#' @return An Inception V3 model, with weights pre-trained on ImageNet.
#' @export
#'
#' @examples
load_imagenet_model <- function(...){

  # load pre-built inception V3 model
  message('\nLoading Model...')
  model <- keras::application_inception_v3(

    weights = "imagenet",
    include_top = FALSE,
    ...
  )
}

#' Calculate Loss & Gradients
#'
#' @param model
#' @param layer_contributions
#'
#' @return
#' @export
#'
#' @examples
calc_loss_and_grads <- function(model, layer_contributions){

  # Set some initial parameters
  k_set_learning_phase(0)
  tf$compat$v1$disable_eager_execution()

  message('\nCalculating loss and gradients...\n')
  # Define loss to be maximised
  layer_dict <- model$layers
  names(layer_dict) <- lapply(layer_dict, function(layer) layer$name)
  loss <- k_variable(0)

  for (layer_name in names(layer_contributions)) {
    coeff <- layer_contributions[[layer_name]]
    activation <- layer_dict[[layer_name]]$output
    scaling <- k_prod(k_cast(k_shape(activation), "float32"))
    loss <- loss + (coeff * k_sum(k_square(activation)) / scaling)
  }

  # Gradient-ascent process
  dream <- model$input
  grads <- k_gradients(loss, dream)[[1]]
  grads <- grads / k_maximum(k_mean(k_abs(grads)), 1e-7)
  outputs <- list(loss, grads)

  k_function(list(dream), outputs)

}

#' Dreamify an Image
#'
#' @param base_image_path
#' @param layer_contributions
#' @param step
#' @param num_octave
#' @param octave_scale
#' @param iterations
#' @param max_loss
#' @param output_image_path
#' @param loss_and_grads
#'
#' @return
#' @export
#'
#' @import tensorflow
#' @import keras
#' @import magrittr
#'
#' @examples
deep_dreamify <- function(

  base_image_path,
  output_image_path,

  loss_and_grads = calc_loss_and_grads(
    model = load_imagenet_model(),
    layer_contributions = list(
      mixed2 = 0.2,
      mixed3 = 3,
      mixed4 = 2,
      mixed5 = 1.5
    )
  ),

  step = 0.01,
  num_octave = 3,
  octave_scale = 1.4,
  iterations = 20,
  max_loss = 10

){

  library(keras)
  library(tensorflow)

  message('\nProcessing Input Image...\n')
  img <- preprocess_image(base_image_path)
  original_shape <- dim(img)[-1]
  successive_shapes <- list(original_shape)

  for (i in 1:num_octave) {
    shape <- as.integer(original_shape / (octave_scale ^ i))
    successive_shapes[[length(successive_shapes) + 1]] <- shape
  }

  successive_shapes <- rev(successive_shapes)
  original_img <- img
  shrunk_original_img <- resize_img(img, successive_shapes[[1]])
  end_shape <- successive_shapes[[length(successive_shapes)]]

  message('\nDreamifying...\n')
  for (shape in successive_shapes) {
    cat("Processsing image shape", shape, "\n")

    img <- resize_img(img, shape) %>%
      gradient_ascent(
        loss_and_grads = loss_and_grads,
        iterations = iterations,
        step = step,
        max_loss = max_loss
      )

    upscaled_shrunk_original_img <- resize_img(shrunk_original_img, shape)
    same_size_original <- resize_img(original_img, shape)
    lost_detail <- same_size_original - upscaled_shrunk_original_img
    img <- img + lost_detail
    shrunk_original_img <- resize_img(original_img, shape)

    # If we're at the end shape, save the new image
    if(identical(end_shape, shape)) save_img(img, fname = output_image_path)

  }

}

