create_ice_plan <- function(lake_ids){
  pb0_ice <- create_task_step(
    step_name = 'ice_flag',
    target_name = function(task_name, step_name, ...) {
      sprintf('tmp/%s_%s.csv', task_name, step_name)
    },

    command = function(task_name, step_name, ...) {
      sprintf("ice_from_GLM_feather(target_name, min_date = I('1980-04-02'),
      '../fig_3/in/%s_temperatures.feather')", task_name)
    }
  )
  create_task_plan(lake_ids$site_id, list(pb0_ice), add_complete = FALSE)
}

create_historical_predict_plan <- function(lake_ids){


  pgdl_predict <- create_task_step(
    step_name = 'predict_pgdl',
    target_name = function(task_name, step_name, ...) {
      sprintf('tmp/%s_%s.csv', task_name, step_name)
    },

    command = function(task_name, step_name, ...) {
      id_only <- strsplit(task_name, '[_]')[[1]][2]
      sprintf("combine_jared_feathers(target_name,
      '../fig_3/in/%sPGRNN_output_trial0.feather',
      '../fig_3/in/%sPGRNN_output_trial1.feather',
      '../fig_3/in/%sPGRNN_output_trial2.feather',
      '../fig_3/in/%sPGRNN_output_trial3.feather',
      '../fig_3/in/%sPGRNN_output_trial4.feather')", id_only, id_only, id_only, id_only, id_only)
    }
  )
  dl_predict <- create_task_step(
    step_name = 'predict_dl',
    target_name = function(task_name, step_name, ...) {
      sprintf('tmp/%s_%s.csv', task_name, step_name)
    },

    command = function(task_name, step_name, ...) {
      id_only <- strsplit(task_name, '[_]')[[1]][2]
      sprintf("combine_jared_feathers(target_name,
      '../fig_3/in/%sRNN_output_trial0.feather',
      '../fig_3/in/%sRNN_output_trial1.feather',
      '../fig_3/in/%sRNN_output_trial2.feather',
      '../fig_3/in/%sRNN_output_trial3.feather',
      '../fig_3/in/%sRNN_output_trial4.feather')", id_only, id_only, id_only, id_only, id_only)
    }
  )

  pb0_predict <- create_task_step(
    step_name = 'predict_pb0',
    target_name = function(task_name, step_name, ...) {
      sprintf('tmp/%s_%s.csv', task_name, step_name)
    },

    command = function(task_name, step_name, ...) {
      sprintf("glm_feather_to_csv(target_name,
      '../fig_3/in/%s_temperatures.feather')", task_name)
    }
  )

  pb_predict <- create_task_step(
    step_name = 'predict_pb',
    target_name = function(task_name, step_name, ...) {
      sprintf('tmp/%s_%s.csv', task_name, step_name)
    },

    command = function(task_name, step_name, ...) {
      sprintf("glm_feather_to_csv(target_name,
      '../fig_3/in/%s_all_profiles_temperatures.feather')", task_name)
    }
  )

  create_task_plan(lake_ids$site_id, list(pgdl_predict, dl_predict, pb0_predict, pb_predict), add_complete = FALSE)
}

create_historical_predict_makefile <- function(makefile, task_plan, final_targets){
  include <- "5_model_predictions.yml"
  packages <- c('dplyr','feather')
  sources <- 'src/task_utils.R'

  create_task_makefile(task_plan, makefile, include = include,
                       packages = packages, sources = sources,
                       final_targets = final_targets)
}

create_ice_makefile <- function(makefile, task_plan, final_targets, finalize_funs){
  include <- "3_model_inputs.yml"
  packages <- c('dplyr','feather')
  sources <- 'src/task_utils.R'

  create_task_makefile(task_plan, makefile, include = include,
                       packages = packages, sources = sources,
                       final_targets = final_targets,
                       finalize_funs = finalize_funs)
}
