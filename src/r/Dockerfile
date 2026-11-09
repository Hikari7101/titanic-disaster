FROM rocker/r-ver:4.4.1

WORKDIR /app

COPY install_packages.R .
RUN Rscript install_packages.R

COPY titanic.R .

CMD ["Rscript", "titanic.R"]