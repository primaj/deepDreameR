resize_img <- function(img, size) {
  image_array_resize(img, size[[1]], size[[2]])
}

save_img <- function(img, fname) {
  img <- deprocess_image(img)
  image_array_save(img, fname)
}

preprocess_image <- function(image_path) {
  image_load(image_path) %>%
    image_to_array() %>%
    array_reshape(dim = c(1, dim(.))) %>%
    inception_v3_preprocess_input()
}

deprocess_image <- function(img) {
  img <- array_reshape(img, dim = c(dim(img)[[2]], dim(img)[[3]], 3))
  img <- img / 2
  img <- img + 0.5
  img <- img * 255
  dims <- dim(img)
  img <- pmax(0, pmin(img, 255))
  dim(img) <- dims
  img
}



#' Pad & Resize an Image
#'
#' Function to pad and resize an image to 300x300, this function currently assumes
#' that the image is longer than it wide and pads the x axes so that it is as long as wide. Subsequently,
#' the image is resized.
#'
#' @param image_path
#' @param size_x
#' @param size_y
#' @param new_path
#'
#' @return
#' @export
#' @importFrom imager load.image pad resize save.image
#' @importFrom stringr str_split
#' @importFrom magrittr extract extract2
#'
#'
#' @examples
pad_and_resize_img <- function(image_path, size_x = 300, size_y = 300,
                               new_path = NULL){

  img <- load.image(image_path)
  dim <- dim(img)

  if(is.null(new_path)) {

    new_path <- sprintf(
      '%s_%sx%s.png',
      str_split(image_path, '\\.') %>% extract2(1) %>% extract(1),
      size_x, size_y
    )
  }

  pad(img, dim[2] - dim[1], axes = 'x') %>%
    resize(size_x, size_y) %>%
    save.image(new_path)

  message(paste0('File saved to:\n', new_path))

}
