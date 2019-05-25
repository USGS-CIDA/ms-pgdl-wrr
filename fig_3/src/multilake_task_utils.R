create_multilake_data_plan <- function(nhd_ids, start, stop, n_profiles_train){
  step1 <- create_task_step(
    step_name = 'get_write_driver',
    target_name = function(task_name, step_name, ...) {
      sprintf('fig_3/VM_shared_cache/%s_meteo.csv', task_name)
    },
    command = function(target_name, task_name, ...) {
      sprintf("get_write_driver(driver = target_name,\n      I('%s'))", task_name, task_name)
    }
  )

  step2 <- create_task_step(
    step_name = 'get_set_nml',
    target_name = function(task_name, step_name, ...) {
      sprintf('fig_3/VM_shared_cache/%s_nml.nml', task_name)
    },
    command = function(target_name, task_name, ...) {
      sprintf("get_write_base_nml(target_name, I('%s'),\n      driver = 'fig_3/VM_shared_cache/%s_meteo.csv',\n      start = I('%s'),\n      stop = I('%s'),\n      dt = I(3600),\n      nsave=I(24))", task_name, task_name, start, stop)
    }
  )

  step3 <- create_task_step(
    step_name = 'export_geometry',
    target_name = function(task_name, step_name, ...) {
      sprintf('fig_3/VM_shared_cache/%s_geometry.csv', task_name)
    },
    command = function(target_name, task_name, ...) {
      sprintf("export_geometry(target_name, I('%s'))", task_name)
    }
  )

  step4 <- create_task_step(
    step_name = 'subset_training_random_01',
    target_name = function(task_name, step_name, ...) {
      exp_n <- strsplit(step_name, '[_]') %>% .[[1]] %>% tail(1)
      prof_n <- stringr::str_pad(n_profiles_train, width = 3, pad = "0")
      sprintf('fig_3/VM_shared_cache/%s_train_%s_profiles_experiment_%s.feather', task_name, prof_n, exp_n) #nhd_4250588_train_010_profiles_experiment_01.feather
    },
    command = function(target_name, task_name, ...) {
      sprintf("subset_training_random(target_name, refined_obs, %s_pretraining_mask)", task_name)
    }
  )

  step5 <- create_task_step(
    step_name = 'subset_training_random_02',
    target_name = function(task_name, step_name, ...) {
      exp_n <- strsplit(step_name, '[_]') %>% .[[1]] %>% tail(1)
      prof_n <- stringr::str_pad(n_profiles_train, width = 3, pad = "0")
      sprintf('fig_3/VM_shared_cache/%s_train_%s_profiles_experiment_%s.feather', task_name, prof_n, exp_n) #nhd_4250588_train_010_profiles_experiment_01.feather
    },
    command = function(target_name, task_name, ...) {
      sprintf("subset_training_random(target_name, refined_obs, %s_pretraining_mask)", task_name)
    }
  )

  step6 <- create_task_step(
    step_name = 'subset_training_random_03',
    target_name = function(task_name, step_name, ...) {
      exp_n <- strsplit(step_name, '[_]') %>% .[[1]] %>% tail(1)
      prof_n <- stringr::str_pad(n_profiles_train, width = 3, pad = "0")
      sprintf('fig_3/VM_shared_cache/%s_train_%s_profiles_experiment_%s.feather', task_name, prof_n, exp_n) #nhd_4250588_train_010_profiles_experiment_01.feather
    },
    command = function(target_name, task_name, ...) {
      sprintf("subset_training_random(target_name, refined_obs, %s_pretraining_mask)", task_name)
    }
  )

  step7 <- create_task_step(
    step_name = 'subset_training',
    target_name = function(task_name, step_name, ...) {
      sprintf('fig_3/VM_shared_cache/%s_train_all_profiles.feather', task_name)
    },
    command = function(target_name, task_name, ...) {
      sprintf("subset_training(target_name, refined_obs, %s_pretraining_mask)", task_name)
    }
  )

  step8 <- create_task_step(
    step_name = 'subset_testing',
    target_name = function(task_name, step_name, ...) {
      sprintf('fig_3/VM_shared_cache/%s_test_all_profiles.feather', task_name)
    },
    command = function(target_name, task_name, ...) {
      sprintf("subset_testing(target_name, refined_obs, %s_pretraining_mask)", task_name)
    }
  )

  step9 <- create_task_step(
    step_name = 'calc_test_masks',
    target_name = function(task_name, step_name, ...) {
      sprintf('%s_pretraining_mask', task_name)
    },
    command = function(target_name, task_name, ...) {
      sprintf("calc_test_masks(refined_obs,\n      site_id = I('%s'),\n      test_location = I(1/3))", task_name)
    }
  )


  task_plan <- create_task_plan(task_names = nhd_ids, list(step1, step2, step3, step4, step5, step6, step7, step8, step9),
                                final_steps='calc_test_masks', add_complete = FALSE)
  return(task_plan)
}

create_multilake_model_plan <- function(nhd_ids, experiment = "random_01", n_profiles_train, sheets_id){

  step1_name <- sprintf('optim_glm_subset_%s', experiment)

  step1 <- create_task_step(
    step_name = step1_name,
    target_name = function(task_name, step_name, ...) {
      sprintf('%s_%s', task_name, step_name)
    },
    command = function(target_name, task_name, step_name, ...) {
      exp_n <- strsplit(step_name, '[_]') %>% .[[1]] %>% tail(1)
      prof_n <- stringr::str_pad(n_profiles_train, width = 3, pad = "0")
      train_file <- sprintf('/media/sf_VM_shared_cache/%s_train_%s_profiles_experiment_%s.feather', task_name, prof_n, exp_n)
      test_file <- sprintf('/media/sf_VM_shared_cache/%s_test_all_profiles.feather', task_name)
      nml_file <- sprintf('/media/sf_VM_shared_cache/%s_nml.nml', task_name)
      driver_file <- sprintf('/media/sf_VM_shared_cache/%s_meteo.csv', task_name)
      sprintf("run_optim_glm(driver_file = '%s',\n      nml_file = '%s',\n      train_file = '%s',\n      test_file = '%s',\n      sheets_id = %s)",
              driver_file, nml_file, train_file, test_file, sheets_id)
    }
  )
  task_plan <- create_task_plan(task_names = nhd_ids, list(step1),
                                final_steps = step1_name, add_complete = FALSE)
  return(task_plan)
}
