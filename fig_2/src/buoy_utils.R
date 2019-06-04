

me_buoy_data <- function(buoy_filepath, discrete_filepath, date_range){

  buoy_temp <- readr::read_csv(buoy_filepath, skip = 1) %>%
    filter(is.na(flag_wtemp) | flag_wtemp == 'N',
           sampledate < as.Date('2009-07-03') | sampledate > as.Date('2009-07-06'),
           sampledate < as.Date('2014-06-17') | sampledate > as.Date('2014-07-10'),
           sampledate < as.Date('2008-11-11') | sampledate > as.Date('2008-12-12'),
           sampledate < as.Date('2014-07-06') | sampledate > as.Date('2014-10-20'),
           sampledate < as.Date('2011-08-07') | sampledate > as.Date('2011-08-19'),
           sampledate < as.Date('2012-08-31') | sampledate > as.Date('2012-09-04')) %>%
    select(DateTime = sampledate, Depth = depth, temp = wtemp)

  manual_temp <- readr::read_csv(discrete_filepath, skip = 1) %>%
    filter(is.na(flagwtemp), !is.na(wtemp)) %>%
    group_by(sampledate, depth) %>% filter(row_number(wtemp) == 1) %>%
    select(DateTime = sampledate, Depth = depth, temp = wtemp)

  combined <- full_join(buoy_temp, manual_temp, by = c("DateTime", "Depth")) %>%
    mutate(temp = ifelse(is.na(temp.x), temp.y, temp.x)) %>% select(-temp.x, -temp.y) %>%
    arrange(DateTime, Depth)

  incomplete_profiles <- combined %>% group_by(DateTime) %>% tally() %>% filter(n < 3) %>% pull(DateTime)

 combined %>% filter(!DateTime %in% incomplete_profiles) %>%
   filter(DateTime >= as.Date(date_range[1]) & DateTime <= as.Date(date_range[2]))
}

sp_buoy_data <- function(buoy_hashed_filepaths, discrete_filepath, date_range){

  get_depth <- function(str){
    depth <- rep(NA_integer_, length(str))
    for (i in 1:length(str)){
      depth[i] <- round(as.numeric(strsplit(str[i], split = '[_]')[[1]][2]), digits = 1)
    }
    return(depth)
  }

  sp_water_temp <- data.frame(time = c(), Depth = c(), temp = c(), stringsAsFactors = FALSE)

  for (file in names(buoy_hashed_filepaths)){
    dat <- readr::read_tsv(file) %>%
      gather(key = "str_depth", value = "temperature", -datetime) %>%
      mutate(Depth = get_depth(str_depth), time = lubridate::as_date(datetime)) %>%
      group_by(time, Depth) %>%
      summarise(temp = mean(temperature, na.rm = TRUE)) %>% data.frame
    sp_water_temp <- rbind(dat, sp_water_temp)
  }

  dat <- readr::read_csv(discrete_filepath) %>% select(time = sampledate, Depth = depth, temp = wtemp)
  rbind(sp_water_temp, dat) %>% rename(DateTime = time) %>%
    filter(DateTime >= as.Date(date_range[1]) & DateTime <= as.Date(date_range[2]))
}


filter_doy <- function(buoy_data, include = NULL, exclude = NULL){

  doy_data <- mutate(buoy_data, doy = lubridate::yday(DateTime))
  if(!is.null(include)){
    doy_data <- filter(doy_data, doy %in% include[1]:include[2])
  }
  if(!is.null(exclude)){
    doy_data <- filter(doy_data, !doy %in% exclude[1]:exclude[2])
  }
  select(doy_data, -doy)
}

filter_year <- function(buoy_data, include = NULL, exclude = NULL){

  year_data <- mutate(buoy_data, year = lubridate::year(DateTime))
  if(!is.null(include)){
    year_data <- filter( year_data, year %in% include)
  }
  if(!is.null(exclude)){
    year_data <- filter( year_data, !year %in% exclude)
  }
  select(year_data, -year)
}
