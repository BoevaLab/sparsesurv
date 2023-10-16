# Process FCS files and Create AnnData Object

## Introduction

In this short tutorial we show how to process fcs files and create
an [anndata object](https://anndata.readthedocs.io/en/latest/). The fcs
files are from the publication _Data-driven phenotypic dissection of AML
reveals progenitor-like cells that correlate with prognosis
(10.1016/j.cell.2015.05.047)_. To facilitate this tutorial we deposited
the required files on
[figshare](https://figshare.com/account/home#/projects/140062).

## Setup

    knitr::opts_chunk$set(message = FALSE, warning = FALSE)
    pkgs = c('flowCore', 'anndata', 'tidyverse', 'magrittr', 'httr', 'reticulate')
    for(i in pkgs){
      suppressPackageStartupMessages(library(i, character.only = TRUE))
    }

## Define the python interpreter to use

Since the AnnData object is a python data structure we need to setup our
python interpreter first. Either you can setup a new environment or use
the one we use for the scQUEST package ([installation guide for
scQUEST](https://ai4scr.github.io/scQUEST/installation.html#package-installation)).
Important is, that the anndata versions of the environment you use here
is the same as in the environment in which you want to use the created
anndata object.

To find the path to the python interpreter run the following snipped in
the terminal

    conda activate scQUEST
    which python
    # /usr/local/Caskroom/miniconda/base/envs/scQUEST/bin/python

Now point `reticulate` to the interpreter you want to use.

    reticulate::use_python('/usr/local/Caskroom/miniconda/base/envs/scQUEST/bin/python')

## Request for figshare api

Here we setup the http request to download the fcs files

    # request for figshare api
    project_id = 140062
    articel_id = 19867114
    PATH = '~/.scQUEST/Levine_CyTOF_AML_AnnotatedCellTypes/'
    REQUEST = paste0('https://api.figshare.com/v2/articles/', articel_id, '/files')

## Helpers

Here we define some helper functions that facilitate mapping of
attributes and the download of the files.

    # helpers
    mapper = function(key, df){
      mapping= list()
      cols = colnames(df)
      cols = cols[cols != key]
      for(i in cols){
        m = map(df[[i]], ~ .x)
        names(m) = df[[key]]
        mapping[[i]] = m
      }
      return (mapping)
    }

    download_files = function(path = PATH, request = REQUEST){
      path = path.expand(path)
      dir.create(path, showWarnings = TRUE, recursive = TRUE, mode = "0777")

      resp = GET(request)
      stop_for_status(resp)

      files = content(resp, 'parsed')
      files = map(files, function(x){
        url = x$download_url
        fpath = paste0(path, x$name)
        if(!file.exists(fpath)){
          download.file(url, fpath)
          }
        return(c(url, fpath))
      })
    }

## Download and mappings

In a next step we download the files form figshare to `PATH`.
Furthermore, we provide some files that provide some more information
regarding the channel names and the sample_ids. For example
`rename.channel` indicates for what this channel was used and what kind
of marker it represents.

    # download files
    files = download_files(PATH, REQUEST)
    files = unlist(map(files, ~.x[2]))
    files.fcs = files[grepl('.fcs$', files)]
    files.csv = files[grepl('.csv$', files)]

    files.sampleID = files.csv[grepl('sampleID', files.csv)]
    files.channels = files.csv[grepl('channels', files.csv)]

    # mappings
    rename.sample = readr::read_csv(files.sampleID)
    map.sample = mapper('sample_id', rename.sample)

    rename.channel = readr::read_csv(files.channels)
    map.channel = mapper('channel', rename.channel)

## Process the fcs files

Now we are ready to process the actuall FCS files. We loop over all fcs
files and construct the attributes we later use to generated the anndata
object (`X`, `OBS`, `VAR`).

    # load FCS file
    X = NULL
    OBS = NULL
    VAR = NULL
    fcs.header = list()

    f.count = 0
    for(f in files.fcs){
      f.count = f.count + 1
      f.base =  basename(f)
      cat(f.count, '/', length(files.fcs), ' reading in file ', f.base, '\n')

      header = read.FCSheader(f)
      header.tbl = tibble(keyword = names(header[[1]]), value=header[[1]])

      fcs <- read.FCS(f)
      x = exprs(fcs)

      obs = list(fcs_file = f.base)
      for(i in names(map.sample)){
        obs[[i]] = map.sample[[i]][[f.base]]
      }

      obs = do.call(tibble, obs)
      obs %<>% slice(rep(1, each = dim(x)[1]))

      var = parameters(fcs)@data %>% select(-range, -minRange, -maxRange)

      X = rbind(X,x)
      OBS = rbind(OBS, obs)

      if(is.null(VAR)) VAR = var
      else stopifnot(all(VAR == var))

      stopifnot(!f %in% names(fcs.header))
      fcs.header[[f.base]] = header.tbl
    }

    ## 1 / 14  reading in file  2011-08-19-AML08-f.A_cct_subtracted_normalized_CD4 T cells_H1.fcs
    ## 2 / 14  reading in file  2011-08-19-AML08-f.A_cct_subtracted_normalized_CD8 T cells_H1.fcs
    ## 3 / 14  reading in file  2011-08-19-AML08-f.A_cct_subtracted_normalized_CD16- NK cells_H1.fcs
    ## 4 / 14  reading in file  2011-08-19-AML08-f.A_cct_subtracted_normalized_CD16+ NK cells_H1.fcs
    ## 5 / 14  reading in file  2011-08-19-AML08-f.A_cct_subtracted_normalized_Mature B cells_H1.fcs
    ## 6 / 14  reading in file  2011-08-19-AML08-f.A_cct_subtracted_normalized_Monocytes_H1.fcs
    ## 7 / 14  reading in file  2011-08-19-AML08-f.A_cct_subtracted_normalized_Pre B cells_H1.fcs
    ## 8 / 14  reading in file  2011-08-25-AML09-a.A_cct_subtracted_normalized_CD4 T cells_H2.fcs
    ## 9 / 14  reading in file  2011-08-25-AML09-a.A_cct_subtracted_normalized_CD8 T cells_H2.fcs
    ## 10 / 14  reading in file  2011-08-25-AML09-a.A_cct_subtracted_normalized_CD16- NK cells_H2.fcs
    ## 11 / 14  reading in file  2011-08-25-AML09-a.A_cct_subtracted_normalized_CD16+ NK cells_H2.fcs
    ## 12 / 14  reading in file  2011-08-25-AML09-a.A_cct_subtracted_normalized_Mature B cells_H2.fcs
    ## 13 / 14  reading in file  2011-08-25-AML09-a.A_cct_subtracted_normalized_Monocytes_H2.fcs
    ## 14 / 14  reading in file  2011-08-25-AML09-a.A_cct_subtracted_normalized_Pre B cells_H2.fcs

    # tidy data
    VAR %<>% rename(channel = name, marker = desc)
    for(i in names(map.channel)){
     VAR %<>% mutate("{i}" := unlist(map.channel[[i]][channel]))
    }

    # create AnnData object
    ad = AnnData(
      X = X,
      var = VAR,
      obs = OBS,
      uns = list(
        fcs_header = fcs.header
      )
    )

    # save AnnData object
    path_output = path.expand(paste0(PATH, 'ad_annotated_cell_types.h5ad'))
    anndata::write_h5ad(ad, path_output)

## Load AnnData object in python

We now created the AnnData object and can use it in our python project

    import anndata
    from pathlib import Path

    fpath = Path(r.path_output)
    assert fpath.is_file()
    ad = anndata.read_h5ad(fpath)
    print(ad)

    ## AnnData object with n_obs × n_vars = 96381 × 39
    ##     obs: 'fcs_file', 'celltype', 'complexcelltype', 'patient', 'ncells'
    ##     var: 'channel', 'marker', 'usedformanualannotation', 'usedforPhenoGraphclustering'
    ##     uns: 'fcs_header'
