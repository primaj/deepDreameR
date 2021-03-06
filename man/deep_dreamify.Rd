% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/dreamify.R
\name{deep_dreamify}
\alias{deep_dreamify}
\title{Dreamify an Image}
\usage{
deep_dreamify(
  base_image_path,
  output_image_path,
  loss_and_grads = calc_loss_and_grads(model = load_imagenet_model(),
    layer_contributions = list(mixed2 = 0.2, mixed3 = 3, mixed4 = 2, mixed5 = 1.5)),
  step = 0.01,
  num_octave = 3,
  octave_scale = 1.4,
  iterations = 20,
  max_loss = 10
)
}
\arguments{
\item{loss_and_grads}{}
}
\value{

}
\description{
Dreamify an Image
}
