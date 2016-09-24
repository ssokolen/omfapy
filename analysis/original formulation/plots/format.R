# Local and extended simulations for original model formulation

library(dplyr)
library(ggplot2)
library(grid)
library(gridExtra)
library(RSQLite)
library(stringr)
library(tidyr)

# Long calculations on database information is cached
OVERWRITE <- FALSE

# Uncomment the following to recalculate intermediate data
#OVERWRITE <- TRUE

# Function for extracting legend
g_legend<-function(a.gplot){
    tmp <- ggplot_gtable(ggplot_build(a.gplot))
    leg <- which(sapply(tmp$grobs, function(x) x$name) == "guide-box")
    legend <- tmp$grobs[[leg]]
    legend
}
    
#---------------------------------------------------------------------
# Local simulation

# Connecting to database
drv <- dbDriver('SQLite')
con <- dbConnect(drv, '../local_simulation.db')

if (OVERWRITE || (! 'local_simulation.Rdata' %in% list.files())) {

  d <- read.csv('../observed_mfa.csv', row.names = 1,
                stringsAsFactors = FALSE)
  d$flux <- rownames(d)

  # Calculating rank to order flux factor
  flux <- d %>% 
            mutate(rank = rank(abs(predicted), ties = 'first')) %>%
            select(flux, rank)

  # Simulated
  query <- "SELECT sample, label, observation, 
                   flux, value, p_t, ci_low, ci_high
            FROM gls_calculated;"

  d <- dbGetQuery(con, query)

  # Loading flux categories
  cols <- c('designation', 'category', 'path')
  names <- read.csv('../../reaction_names.csv', stringsAsFactors = FALSE)[, cols]
  colnames(names) <- c('flux', 'group', 'path')
  names$row <- 1:nrow(names)

  d <- d %>%
         left_join(flux, by = 'flux') %>%
         left_join(names, by = 'flux') %>%
         arrange(row) %>%
         mutate(group = factor(group, levels = unique(group))) %>%
         arrange(rank) %>%
         mutate(label = ifelse(grepl('_\\d', flux), path, flux),
                label = factor(label, levels = unique(label)))

  # All the data
  simulated <- d

  # Summarizing
  simulated.stats <- d %>%
                       group_by(flux) %>%
                       summarize(group = group[1], 
                                 label = label[1], 
                                 rejected = sum(p_t > 0.05)/n()*100,
                                 value = median(value))

  save(simulated, simulated.stats, file='local_simulation.Rdata')
} else {
  load('local_simulation.Rdata')
}

# Adding observed values
observed <- read.csv('../observed_mfa.csv', row.names = 1,
                     stringsAsFactors = FALSE)
observed$flux <- rownames(observed)
observed <- select(observed, calculated = predicted, flux, p_t)

simulated.stats <- simulated.stats %>%
                     left_join(observed, by = 'flux')

# Flux value
p1 <- ggplot(simulated.stats, aes(y = label, x = abs(calculated), 
                                  shape = group, fill = group))
p1 <- p1 + geom_point(size = 4)

p1 <- p1 + scale_x_log10(breaks = c(0.1, 1, 10, 100))
shape.values <- c(21, 22, 23, 21, 
                  22, 23, 24)
p1 <- p1 + scale_shape_manual('Flux category', values = shape.values,
                              guide = guide_legend(nrow = 2, 
                                                   byrow = TRUE))
fill.values <- c('black', 'black', 'black', 'white', 
                 'white', 'white', 'white')
p1 <- p1 + scale_fill_manual('Flux category', values = fill.values,
                              guide = guide_legend(nrow = 2, 
                                                   byrow = TRUE))
p1 <- p1 + xlab(expression(paste(
                  'Flux ',
                  scriptstyle(bgroup(
                    '(', 
                    frac('nmol', paste(10^6, ' cells' %.% 'h')),
                    ')'))))) 
p1 <- p1 + ylab('')

p1 <- p1 + theme_bw(16)

col <- rgb(1, 1, 1, alpha = 0)
p1 <- p1 + theme(axis.text.y = element_text(size = 12),
                 axis.text.x = element_text(size = 16),
                 axis.title.x = element_text(size = 16),
                 legend.position = 'top',
                 legend.key = element_rect(size = 5, colour = col),
                 legend.key.size = unit(1.5, 'lines'))

legend <- g_legend(p1)

p1 <- p1 + theme(legend.position = 'none')

# Percent rejected
p2 <- ggplot(simulated.stats, aes(x = label, y = rejected))
p2 <- p2 + geom_bar(stat='identity', position='identity')

y.point <- max(simulated.stats$rejected) * 1.05
p2 <- p2 + geom_point(data = filter(simulated.stats, p_t > 0.05),
                      aes(x = label, y = y.point), shape = 8)

p2 <- p2 + ylab(expression(paste(
                  'Percent non-significant (%)',
                  phantom(scriptstyle(bgroup('(', frac(1,1^1), ')')))))) 
p2 <- p2 + xlab('')

p2 <- p2 + theme_bw(16)
p2 <- p2 + coord_flip()

p2 <- p2 + theme(axis.text.y = element_blank(),
                 axis.text.x = element_text(size = 16),
                 axis.title.x = element_text(size = 16),
                 axis.ticks.y = element_blank())

# Combining
pdf('local_simulation.pdf', width = 12, height = 8)
  grid.arrange(legend,
               arrangeGrob(p1, p2, widths = c(0.6, 0.4)),
               heights = c(0.07, 0.93))

  par <- gpar(fontsize = 20)
  grid.text('A', x = unit(0.24, 'npc'), y = unit(0.90, 'npc'), gp = par)
  grid.text('B', x = unit(0.61, 'npc'), y = unit(0.90, 'npc'), gp = par)
dev.off()

dbDisconnect(con)

#------------------------------------------------------------------------
# Extended simulation

# Connecting to database
drv <- dbDriver('SQLite')
con <- dbConnect(drv, '../extended_simulation.db')

if (OVERWRITE || (! 'extended_simulation.Rdata' %in% list.files())) {

  d_raw <- dbGetQuery(con, "SELECT g.label, g.sample, g.observation, 
                                   g.flux, g.p_t, g.value as observed, 
                                   s.value as real, s.rank
                            FROM gls_calculated as g
                            LEFT JOIN samples as s
                            ON s.sample = g.sample AND
                               s.flux = g.flux
                            LEFT JOIN pi_overall as p
                            ON p.sample = g.sample AND
                               p.label = g.label AND
                               p.observation = g.observation
                            WHERE s.observed = 0 AND
                                  p_chi2 > 0.05")

  # Resetting rank and calculating deviations
  d_raw <- d_raw %>%
             mutate(deviation = abs(observed - real) / real * 100)


  # Aggregating data
  d_samples <- d_raw %>%
                 group_by(sample, flux) %>%
                 summarize(real = abs(real[1]), rank = rank[1])

  d_samples <- d_samples %>%
                 group_by(sample) %>%
                 arrange(real) %>%
                 mutate(rank = 1:n())

  d_raw <- left_join(select(d_raw, -rank), 
                     select(d_samples, sample, flux, rank),
                     by = c('sample', 'flux'))

  labels <- unique(d_raw$label)
  d_raw$label <- factor(d_raw$label, 
                        levels=sort(labels, decreasing=TRUE))

  d_observations <- d_raw %>%
                      group_by(label, rank) %>%
                      summarize(error = median(abs(deviation)),
                                rejected = sum(p_t > 0.05) / n() * 100)

  save(d_samples, d_observations, file='extended_simulation.Rdata')
} else {
  load('extended_simulation.Rdata')
}

# Formatting y-axis labels so they align
f_pad <- function(l) {
  
  # Base formatting
  l <- format(l, justify = 'right')

  # Padding
  l <- str_pad(l, 5, side = 'left', pad = ' ') 
  
  l
}

# Real flux box plots
p1 <- ggplot(d_samples)
p1 <- p1 + stat_boxplot(aes(x = rank, y = real, group = rank))
p1 <- p1 + scale_y_continuous(trans = 'log10', 
                              breaks = c(0.1, 1, 10, 100), labels = f_pad)
p1 <- p1 + theme_bw(16)
p1 <- p1 + ylab(expression(paste(
            'Flux ',
            scriptstyle(bgroup(
              '(', 
              frac('nmol', paste(10^6, ' cells' %.% 'h')),
              ')'))))) 
p1 <- p1 + xlab('Flux rank')

# Observed flux error
p2 <- ggplot(d_observations)
p2 <- p2 + geom_bar(aes(x = rank, y = error, fill = label),
                    stat = 'identity', position = 'identity')
label <- 'measurement standard deviation (%)'
p2 <- p2 + scale_fill_grey(name = label, start=0.2, end=0.8)
p2 <- p2 + scale_y_continuous(labels = f_pad)
p2 <- p2 + theme_bw(16)
p2 <- p2 + theme(legend.position = 'none')
p2 <- p2 + ylab(expression(paste(
            'Median error (%)',
            phantom(scriptstyle(bgroup('(', frac(1,1^1), ')')))))) 
p2 <- p2 + xlab('Flux rank')

# Percent rejected
p3 <- ggplot(d_observations)
p3 <- p3 + geom_bar(aes(x = rank, y = rejected, fill = label),
                    stat = 'identity', position = 'identity')
label <- 'Measurement standard deviation (%)  '
p3 <- p3 + scale_fill_grey(name = label, start=0.2, end=0.8)
p3 <- p3 + scale_y_continuous(labels = f_pad)
p3 <- p3 + theme_bw(16)
p3 <- p3 + ylab(expression(paste(
            'Fluxes rejected (%)',
            phantom(scriptstyle(bgroup('(', frac(1,1^1), ')')))))) 
p3 <- p3 + xlab('Flux rank')

p3 <-p3 + theme(legend.position = 'top')
legend <- g_legend(p3)
p3 <- p3 + theme(legend.position = 'none')

# Combining
pdf('extended_simulation.pdf', width = 7, height = 8)
  grid.arrange(legend, p1, p2, p3, heights = c(0.05, 0.3, 0.3, 0.3))

  par <- gpar(fontsize = 20)
  grid.text('A', x = unit(0.05, 'npc'), y = unit(0.96, 'npc'), gp = par)
  grid.text('B', x = unit(0.05, 'npc'), y = unit(0.64, 'npc'), gp = par)
  grid.text('C', x = unit(0.05, 'npc'), y = unit(0.32, 'npc'), gp = par)
dev.off()

dbDisconnect(con)
