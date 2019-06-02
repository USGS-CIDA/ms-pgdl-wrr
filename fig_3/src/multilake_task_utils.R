create_multilake_data_plan <- function(nhd_ids, start, stop, n_profiles_train){

  nhd_ids <- unique(nhd_ids)
  prof_n <- stringr::str_pad(n_profiles_train, width = 3, pad = "0")
  step1 <- create_task_step(
    step_name = 'get_write_driver',
    target_name = function(task_name, step_name, ...) {
      sprintf('fig_3/yeti_sync/%s_meteo.csv', task_name)
    },
    command = function(target_name, task_name, ...) {
      sprintf("get_write_driver(driver = target_name,\n      I('%s'))", task_name, task_name)
    }
  )

  step2 <- create_task_step(
    step_name = 'get_set_nml',
    target_name = function(task_name, step_name, ...) {
      sprintf('fig_3/yeti_sync/%s_nml.nml', task_name)
    },
    command = function(target_name, task_name, ...) {
      sprintf("get_write_base_nml(target_name, I('%s'),\n      driver = I('%s_meteo.csv'),\n      start = I('%s'),\n      stop = I('%s'),\n      dt = I(3600),\n      nsave=I(24))", task_name, task_name, start, stop)
    }
  )

  step3 <- create_task_step(
    step_name = 'subset_training_random_01',
    target_name = function(task_name, step_name, ...) {
      exp_n <- strsplit(step_name, '[_]') %>% .[[1]] %>% tail(1)
      sprintf('fig_3/yeti_sync/%s_train_%s_profiles_experiment_%s.csv', task_name, prof_n, exp_n) #nhd_4250588_train_010_profiles_experiment_01.csv
    },
    command = function(target_name, task_name, ...) {
      sprintf("subset_training_random(target_name, refined_obs, %s_pretraining_mask)", task_name)
    }
  )

  step4 <- create_task_step(
    step_name = 'subset_training_random_02',
    target_name = function(task_name, step_name, ...) {
      exp_n <- strsplit(step_name, '[_]') %>% .[[1]] %>% tail(1)
      sprintf('fig_3/yeti_sync/%s_train_%s_profiles_experiment_%s.csv', task_name, prof_n, exp_n) #nhd_4250588_train_010_profiles_experiment_01.csv
    },
    command = function(target_name, task_name, ...) {
      sprintf("subset_training_random(target_name, refined_obs, %s_pretraining_mask)", task_name)
    }
  )

  step5 <- create_task_step(
    step_name = 'subset_training_random_03',
    target_name = function(task_name, step_name, ...) {
      exp_n <- strsplit(step_name, '[_]') %>% .[[1]] %>% tail(1)
      sprintf('fig_3/yeti_sync/%s_train_%s_profiles_experiment_%s.csv', task_name, prof_n, exp_n) #nhd_4250588_train_010_profiles_experiment_01.csv
    },
    command = function(target_name, task_name, ...) {
      sprintf("subset_training_random(target_name, refined_obs, %s_pretraining_mask)", task_name)
    }
  )

  step6 <- create_task_step(
    step_name = 'subset_training',
    target_name = function(task_name, step_name, ...) {
      sprintf('fig_3/yeti_sync/%s_train_all_profiles.csv', task_name)
    },
    command = function(target_name, task_name, ...) {
      sprintf("subset_training(target_name, refined_obs, %s_pretraining_mask)", task_name)
    }
  )

  step7 <- create_task_step(
    step_name = 'subset_testing',
    target_name = function(task_name, step_name, ...) {
      sprintf('fig_3/yeti_sync/%s_test_all_profiles.csv', task_name)
    },
    command = function(target_name, task_name, ...) {
      sprintf("subset_testing(target_name, refined_obs, %s_pretraining_mask)", task_name)
    }
  )

  step8 <- create_task_step(
    step_name = 'calc_test_masks',
    target_name = function(task_name, step_name, ...) {
      sprintf('%s_pretraining_mask', task_name)
    },
    command = function(target_name, task_name, ...) {
      sprintf("calc_test_masks(refined_obs,\n      site_id = I('%s'),\n      test_location = I(1/3))", task_name)
    }
  )

  step9 <- create_task_step(
    step_name = 'indicate_sync_files',
    target_name = function(task_name, step_name, ...) {
      sprintf('%s_sync_files', task_name)
    },
    command = function(target_name, task_name, ...) {
      sprintf("combine_to_ind(ind_file = I(''),\n      'fig_3/yeti_sync/%s_meteo.csv',\n      'fig_3/yeti_sync/%s_nml.nml',
      'fig_3/yeti_sync/%s_train_%s_profiles_experiment_01.csv',\n      'fig_3/yeti_sync/%s_train_%s_profiles_experiment_02.csv',\n      'fig_3/yeti_sync/%s_train_%s_profiles_experiment_03.csv',
      'fig_3/yeti_sync/%s_train_all_profiles.csv',\n      'fig_3/yeti_sync/%s_test_all_profiles.csv')",  task_name, task_name, task_name, prof_n, task_name, prof_n, task_name, prof_n, task_name, task_name)
    }
  )


  task_plan <- create_task_plan(task_names = nhd_ids, list(step1, step2, step3, step4, step5, step6, step7, step8, step9),
                                final_steps='indicate_sync_files', add_complete = FALSE)
  return(task_plan)
}

create_multilake_model_plan <- function(nhd_ids, experiment = "random_01", n_profiles_train, sheets_id){

  nhd_ids <- unique(nhd_ids)
  sheet_anchors <- data.frame(id = nhd_ids) %>% mutate(gs_anchor = sprintf("A%s", row_number()+1))

  step1_name <- sprintf('optim_glm_subset_%s', experiment)

  step2_name <- sprintf('optim_posted_subset_%s', experiment)

  step1 <- create_task_step(
    step_name = step1_name,
    target_name = function(task_name, step_name, ...) {
      sprintf('%s_%s', task_name, step_name)
    },
    command = function(target_name, task_name, step_name, ...) {
      exp_n <- strsplit(step_name, '[_]') %>% .[[1]] %>% tail(1)
      prof_n <- stringr::str_pad(n_profiles_train, width = 3, pad = "0")
      train_file <- sprintf('/media/sf_VM_shared_cache/%s_train_%s_profiles_experiment_%s.csv', task_name, prof_n, exp_n)
      test_file <- sprintf('/media/sf_VM_shared_cache/%s_test_all_profiles.csv', task_name)
      nml_file <- sprintf('/media/sf_VM_shared_cache/%s_nml.nml', task_name)
      driver_file <- sprintf('/media/sf_VM_shared_cache/%s_meteo.csv', task_name)
      sprintf("run_optim_glm(driver_file = '%s',\n      nml_file = '%s',\n      train_file = '%s',\n      test_file = '%s')",
              driver_file, nml_file, train_file, test_file, sheets_id)
    }
  )

  step2 <- create_task_step(
    step_name = step2_name,
    target_name = function(task_name, step_name, ...) {
      sprintf('%s_%s', task_name, step_name)
    },
    command = function(target_name, task_name, step_name, ...) {
      gs_anchor = sheet_anchors %>% filter(id == task_name) %>% pull(gs_anchor)
      sprintf("post_results_sheet(target_name, results_df = %s,\n      sheets_id = %s,\n      gs_anchor = I('%s'))",
              sprintf('%s_%s', task_name, step1_name), sheets_id, gs_anchor)
    }
  )

  task_plan <- create_task_plan(task_names = nhd_ids, list(step1, step2),
                                final_steps = step2_name, add_complete = FALSE)
  return(task_plan)
}

combine_to_job_table <- function(target_name, ...){
  #job_table <- data.frame(site_id = c(), nml_file = c(), meteo_file = c(), exper_id = c(), train_file = c(), test_file = c())
  files <- c(...) %>% names() %>% basename()
  train_files <- files[stringr::str_detect(files, "train")]

  get_site_id <- function(train_file){
    sapply(train_file, FUN = function(x) paste(strsplit(x, '[_]')[[1]][1:2], collapse = '_'), USE.NAMES = FALSE)
  }

  get_x_file <- function(site_id, pattern){
    x_files <- files[stringr::str_detect(files, pattern)]
    files_out <- site_id
    for (i in 1:length(files_out)){
      files_out[i] <- x_files[stringr::str_detect(x_files, sprintf('%s%s', site_id[i], pattern))]
    }
    return(files_out)
  }
  get_exper_id <- function(site_id, train_file){
    exper_id <- site_id
    for (i in 1:length(site_id)){
      details <- strsplit(train_file[i], '[_]')[[1]] %>% tail(-3)
      details[length(details)] <- tail(details, 1) %>% tools::file_path_sans_ext()
      exper_id[i] <- paste(site_id[i], paste(details, collapse = '_'), sep = '_')
    }
    return(exper_id)
  }



  data.frame(train_file = train_files, stringsAsFactors = FALSE) %>%
    mutate(site_id = get_site_id(train_file),
           test_file = get_x_file(site_id, '_test_'),
           nml_file = get_x_file(site_id, '_nml.'),
           meteo_file = get_x_file(site_id, '_meteo.'),
           exper_id = get_exper_id(site_id, train_file)) %>%
    select(site_id, nml_file, meteo_file, exper_id, train_file, test_file) %>%
    readr::write_csv(target_name)

}
