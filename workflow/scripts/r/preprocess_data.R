#' Chooses the proper sample in case donors have multiple primary samples.
#' We first chose the "lower" vial (i.e., we pick vial A over vial B, since
#' the lower vials tend to be much more common). Afterward, if there are still
#' multiple samples for a donor, we follow the official guidance from the broad
#' institute by choosing the lexicographically smaller barcode.
#'
#' @param barcodes vector. Vector containing two or more barcodes
#'                         for the same patient.
#'
#' @returns character. We return only one "chosen" barcode, based on the logic above.
choose_proper_sample <- function(barcodes) {
  vials <- substr(barcodes, 16, 16)
  if (length(unique(vials)) > 1) {
    return(str_sort(barcodes, numeric = TRUE)[1])
  } else {
    return(str_sort(barcodes, numeric = TRUE, decreasing = TRUE)[1])
  }
}

#' Filters TCGA samples based on their barcode. Excludes everything but non-primary
#' samples. Also makes sure that each donor is represented
#' through only one sample (see `choose_proper_sample`).
#'
#'
#' @param df data.frame. data.frame containing the data in question. Colnames must
#'                       be TCGA barcodes, rows are expected to contain molecular features
#'                       (although we don't touch them here).
#'
#' @returns data.frame. data.frame filtered based on the principles above.
#'                      This new data.frame will contain only samples of the
#'                      specified types and additionally contain only one
#'                      sample for each donor.
filter_samples <- function(df) {
  # Get tissue types from barcodes.
  types <- substr(colnames(df), 14, 15)

  # 01 - Primary Solid Tumor
  # 03 - Primary Blood Derived Cancer - Peripheral Blood
  # 09 - Primary Blood Derived Cancer - Bone Marrow

  # 04 - Recurrent Blood Derived Cancer - Bone Marrow	TRBM
  # 06 - Metastatic	TM

  # Select only primary tumors.
  selected_types <- c("01", "03", "09")
  df <- df[, types %in% selected_types]
  # The first 12 characters of the barcode uniquely determine the donor
  # in question.
  donors <- substr(colnames(df), 1, 12)

  # Make sure no donors are duplicated by using the logic detailed in
  # `choose_proper_sample`.
  duplicated_donors <- names(which(table(donors) > 1))
  if (length(duplicated_donors) > 0) {
    unique_df <- df[, !donors %in% duplicated_donors]
    df <- cbind(unique_df, df[, unname(sapply(
      unlist(lapply(duplicated_donors, function(x) choose_proper_sample(grep(x, colnames(df), value = TRUE)))),
      function(x) grep(x, colnames(df))
    ))])
  }
  # Cut barcode to only the patient id for modeling.
  colnames(df) <- unname(sapply(colnames(df), function(x) substr(x, 1, 12)))
  return(df)
}

#' Impute data. We use the same imputation logic for all molecular features.
#' Clinical features are handled separately (see our paper for details).
#'
#' @param df data.frame. data.frame containing some missing values that are to be imputed.
#'
#' @returns data.frame. Complete data.frame for which all missing values have been
#'                      either imputed or the covariates in question have been removed.
impute <- function(df) {
  # Columns contain patients - we exclude patients which are missing for more
  # than 10% of all patients.
  large_missing <- which(apply(df, 1, function(x) sum(is.na(x))) > round(dim(df)[2] / 10))
  if (length(large_missing) > 0) {
    df <- df[-which(apply(df, 1, function(x) sum(is.na(x))) > round(dim(df)[2] / 10)), ]
  }

  # Any missing values still left over after this initial filtering step are
  # imputed using the median value per feature.
  if (any(is.na(df))) {
    for (na_count in which(apply(df, 1, function(x) any(is.na(x))))) {
      df[na_count, ] <- as.numeric(df[na_count, ]) %>% replace_na(median(as.numeric(df[na_count, ]), na.rm = TRUE))
    }
  }
  return(df)
}

#' Performs complete preprocessing of molecular data:
#' - Filtering: Removing non-primary/normal samples and making sure
#'              each donor is represented by only one sample
#' - Imputation: Imputing or removing covariates with missing values
#' - Logging: Some molecular data is best represented in log-space. Thus,
#'            these data are logged here.
#'
#' @param df data.frame. data.frame containing molecular data to be preprocessed.
#' @param log logical. Whether the molecular data should be logged.
#' @returns data.frame. Preprocessed data.frame.
preprocess <- function(df, log = FALSE) {
  if (all(nchar(colnames(df)) >= 15)) {
    df <- filter_samples(df)
  }
  if (any(is.na(df))) {
    df <- impute(df)
  }
  if (log) {
    df <- log(1 + df, base = 2)
  }
  return(df)
}

#' Performs complete preprocessing of TCGA-PANCANATLAS gene expression data.
#'
#' @param gex data.frame. data.frame containing mRNA data to be preprocessed.
#' @returns data.frame. Preprocessed data.frame.
prepare_gene_expression_pancan <- function(gex) {
  rownames(gex) <- gex[, 1]
  gex <- gex[, 2:ncol(gex)]
  gex <- preprocess(gex, log = TRUE)
  return(gex)
}

#' Performs complete preprocessing of TCGA clinical data.
#'
#' @param clinical_raw data.frame. data.frame containing TCGA-CDR clinical data.
#' @param clinical_ext_raw data.frame. data.frame containing TCGA-TSV (clinical with follow-up) clinical data.
#' @param cancer character data.frame. Cancer dataset name to be used.
#' @returns data.frame. Preprocessed data.frame with clinical data.
prepare_clinical_data <- function(clinical_raw, clinical_ext_raw, cancer) {
  # Keep patients that are missing survival information if the user desires.
  clinical <- clinical_raw %>%
    # remove any patients for which the OS endpoint is missing
    filter(!(is.na(OS) | is.na(OS.time))) %>%
    # remove any patients which were not at risk at the start of the study
    filter(!(OS.time == 0))
  # Select out clinical covariates in question and recode
  # all missing data to `NA`.
  clinical <- clinical %>%
    filter(type == cancer) %>%
    dplyr::select(
      bcr_patient_barcode, OS, OS.time
    ) %>%
    rename(patient_id = bcr_patient_barcode)
  return(clinical)
}



preprocess_data <- function(cancer,
                                       tcga_cdr_master,
                                       tcga_w_followup_master,
                                       gex_master,
                                       output_path) {
  clinical <- prepare_clinical_data(tcga_cdr_master, tcga_w_followup_master, cancer = cancer)
  sample_barcodes <- list(clinical$patient_id)
  patients <- unname(unlist(sapply(clinical$patient_id, function(x) grep(x, colnames(gex_master)))))
  gex_filtered <- gex_master[, c(1, patients)]
  gex <- prepare_gene_expression_pancan(gex_filtered)
  sample_barcodes <- append(sample_barcodes, list(colnames(gex)))

  common_samples <- Reduce(intersect, sample_barcodes)
  data <- clinical %>%
    filter(patient_id %in% common_samples) %>%
    arrange(desc(patient_id))
  data <- data %>%
    cbind(
      data.frame(t(gex), check.names = FALSE) %>%
        rownames_to_column() %>%
        filter(rowname %in% common_samples) %>%
        arrange(desc(rowname)) %>%
        dplyr::select(-rowname) %>%
        rename_with(function(x) paste0("gex_", x))
    )

  print(paste0("Writing: ", cancer))
  data %>%
    rename(OS_days = OS.time) %>%
    write_csv(
      output_path
    )
}

rerun_preprocessing_R <- function(gex_path, cdr_path, followup_path, cancer, output_path) {
  # Increase VROOM connection size for larger PANCAN files.
  Sys.setenv("VROOM_CONNECTION_SIZE" = 131072 * 8)

  gex_master <- vroom(
    gex_path
  ) %>% data.frame(check.names = FALSE)

  tcga_cdr <- readxl::read_xlsx(
    cdr_path,
    guess_max = 2500,
    range = cell_cols("B:AH")
  )
  tcga_w_followup <- read_tsv(
    followup_path,
    guess_max = 1e5
  )
  preprocess_data(
    cancer = cancer,
    tcga_cdr_master = tcga_cdr,
    tcga_w_followup_master = tcga_w_followup,
    gex_master = gex_master,
    output_path = output_path
  )
  return(0)
}

log <- file(snakemake@log[[1]], open = "wt")
sink(log)
suppressPackageStartupMessages({
  library(dplyr)
  library(readr)
  library(tidyr)
  library(stringr)
  library(vroom)
  library(readxl)
  library(tibble)
})

rerun_preprocessing_R(
  gex_path = snakemake@input[["gex"]],
  cdr_path = snakemake@input[["cdr"]],
  followup_path = snakemake@input[["followup"]],
  cancer = snakemake@params[["cancer"]],
  output_path = snakemake@output[["output_path"]]
)
