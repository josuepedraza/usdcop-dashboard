# Paquetes 
library(quantmod)
library(dplyr)
library(tseries)
library(ggplot2)
library(forecast)
library(rugarch)
library(gridExtra)
library(FinTS)
library(tibble)

# 1. Datos y construcción de variables

# Calcular retornos logarítmicos diarios
Retornos <- function(precios) {
  n <- nrow(precios)
  ret <- xts(rep(NA, n), order.by = index(precios)) # vector retorno
  
  for (i in 2:n) {
    # buscar el último precio válido hacia atrás
    p <- as.numeric(precios[i])
    p1 <- as.numeric(precios[i-1])
    p2 <- as.numeric(precios[i-2])
    
    if (!is.na(p1)) {
      ret[i] <- log(p)-log(p1)
    } else {
      ret[i] <- log(p)-log(p2)
    }
  }
  
  colnames(ret) <- colnames(precios)
  return(ret)
}

## 1.1 Variable dependiente e independientes
symbols <- c("COP=X", "CL=F", "DX-Y.NYB", "GC=F", "^VIX", "^TNX")

getSymbols(symbols, src = "yahoo",
           from = "2025-06-30", to = "2025-11-12", auto.assign = TRUE)

# Extract close prices
usd_cop <- Cl(`COP=X`)
cl_f    <- Cl(`CL=F`)
dxy     <- Cl(`DX-Y.NYB`)
gc_f    <- Cl(`GC=F`)
vix     <- Cl(`VIX`)
tnx     <- Cl(TNX)
rm(`COP=X`,`CL=F`,`DX-Y.NYB`,`GC=F`,`VIX`,TNX, symbols)

# Compute returns
df <- merge(
  Retornos(usd_cop),
  Retornos(cl_f),
  Retornos(dxy),
  Retornos(gc_f),
  Retornos(vix),
  Retornos(tnx)
)

rm(usd_cop,cl_f,dxy,gc_f,vix,tnx)
colnames(df) <- c("usd_cop", "cl_f", "dxy", "gc_f", "vix", "tnx")
df <- na.omit(df)
head(df)
# 2. Supuestos del modelo

## Grafica endogena

df %>% 
  as.data.frame() %>%
  mutate(date = as.Date(rownames(.))) %>%
  ggplot(aes(x = date)) +
  geom_line(aes(y = usd_cop, color = "usd_cop"), linewidth = 0.4) +
  labs(title = "Retornos del USD/COP 2025", x = "Fecha", y = "Retorno")


## Grafica exogenas

df %>% as.data.frame() %>% mutate(date = as.Date(rownames(.))) %>%
  tidyr::pivot_longer(-date) %>%
  ggplot(aes(x = date, y = value, color = name)) +
  geom_line(linewidth = 0.3) +
  theme_minimal() +
  labs(title="Retornos de Variables Exógenas 2025", x="Fecha", y="Retorno", color="Serie")

## Pruebas de estacionaridad

adf.test(na.omit(df$usd_cop))
adf.test(na.omit(df$cl_f))
adf.test(na.omit(df$dxy))

## Autocorrelación
us <- df$usd_cop
diffus <- na.omit(diff(us,5))
ndiffs(us)

par(mfrow = c(2,2))
acf(us, 40,main="ACF USD/COP")
pacf(us, 40,main="PACF USD/COP")
acf(diffus,40)
pacf(diffus,40)
par(mfrow = c(1,1))

## Modelo Sarimax y Garch

train <- df['/2025-09-30']
y_train <- as.numeric(train$usd_cop)
X_train <- as.matrix(train[, c("cl_f","dxy","tnx","vix")])

car::vif(lm(usd_cop~cl_f+dxy+gc_f+vix+tnx, train))

m1 <- auto.arima(y_train, max.p = 15, max.q = 15, xreg=X_train, stationary = TRUE, seasonal = FALSE, nmodels = 10000)
m2 <- Arima(y_train,  order = c(12, 0, 12),  seasonal = list(order = c(5, 1, 7),period = 5),
            xreg = X_train)
m3 <- Arima(y_train,  order = c(12, 0, 0),  xreg = X_train, include.mean = F)

summary(m3) # Modelo del articulo
summary(m2) # Modelo Propio
summary(m1) # Autoarima

par(mfrow=c(1,2))
acf(residuals(m3)^2,  main="ACF (res^2)")
pacf(residuals(m3)^2, main="PACF (res^2)")
par(mfrow=c(1,1))

# GARCH(1,1)
spec_garch <- ugarchspec(
  variance.model = list(model = "sGARCH", garchOrder = c(2,0)),
  mean.model = list(armaOrder = c(1,1), include.mean = FALSE),
  distribution.model = "std"
)

# ARCH(1)
spec_arch <- ugarchspec(
  variance.model = list(model = "sGARCH", garchOrder = c(1,0)),
  mean.model = list(armaOrder = c(1,0), include.mean = FALSE),
  distribution.model = "std"
)

fit_garch  <- ugarchfit(spec_garch,  data = residuals(m3))
fit_arch <- ugarchfit(spec_arch, data = residuals(m3))

par(mfrow=c(2,2))
acf(residuals(fit_garch)^2, main="ACF GARCH(1,1)")
pacf(residuals(fit_garch)^2, main="PACF GARCH(1,1)")

acf(residuals(fit_arch)^2, main="ACF ARCH(1)")
pacf(residuals(fit_arch)^2, main="PACF ARCH(1)")
par(mfrow=c(1,1))


## Pruebas del modelo


# TRAIN
train <- df['/2025-10-14']
y_train <- as.numeric(train$usd_cop)
X_train <- as.matrix(train[, c("cl_f","dxy","tnx","vix")])

# TEST
test  <- df['2025-10-15/']

y_test  <- as.numeric(test$usd_cop)
X_test  <- as.matrix(test[, c("cl_f","dxy","tnx","vix")])
n_test  <- nrow(X_test)

#### Estimación ####
spec_train <- ugarchspec(
  variance.model = list(
    model = "sGARCH",
    garchOrder = c(1,1)
  ),
  mean.model = list(
    armaOrder = c(12,0),
    include.mean = TRUE,
    external.regressors = X_train
  ),
  distribution.model = "std"
)

fit_train <- ugarchfit(
  spec = spec_train,
  data = y_train,
  out.sample = n_test
)

show(fit_train)

#### Pronostico ###
fc <- ugarchforecast(
  fitORspec = fit_train,
  n.ahead   = 1,
  n.roll    = n_test - 1,
  external.forecasts = list(
    mregfor = X_test,   # exógenas para el periodo de prueba
    vregfor = NULL
  )
)

# Medias pronosticadas (rendimientos) para el TEST
pred_mean  <- as.numeric(tail(fitted(fc), n_test))

# Volatilidad condicional pronosticada (sigma_t)
pred_sigma <- as.numeric(tail(sigma(fc),  n_test))

### Grafica del pronostico
data.frame(
  real = y_test,
  pronostico = pred_mean)%>%
  mutate(tiempo = as.Date(rownames(X_test))) %>%
  ggplot( aes(x = tiempo)) +
  geom_line(aes(y = real, color = "Real"), size = 0.7) +
  geom_line(aes(y = pronostico, color = "Pronóstico"), size = 1, linetype = "dashed") +
  labs(title = "Comparación de series (Real vs Pronóstico)",
       x = "Tiempo",
       y = "Valor",
       color = "Serie") +
  scale_color_manual(values = c("Real" = "black", "Pronóstico" = "red")) +
  theme_minimal(base_size = 14)

#### Medidas Error #### 

errores <- (y_test - pred_mean)*100

RMSE <- sqrt(mean(errores^2))
MAE  <- mean(abs(errores))
MAPE <- mean(abs(errores / y_test))

data.frame(RMSE, MAE, MAPE)


#### Diagnostico residuos 

z <- residuals(fit_train, standardize = TRUE) %>% as.data.frame() %>%
  rownames_to_column()
colnames(z) <- c("Fecha", "Residual")
z$Fecha <- as.Date(rownames(X_train)[-c(42:59)])

# Serie de residuos 
p1 <- ggplot(z, aes(x = Fecha, y = Residual)) +
  geom_line(color = "black", linewidth = 0.4) +
  labs(title = "Residuos estandarizados", x = "Fecha", y = "Residual") +
  theme_minimal()

# ACF de residuos
acf_r <- acf(z$Residual, plot = FALSE)
acf_df <- with(acf_r, data.frame(lag, acf))

p2 <- ggplot(acf_df, aes(x = lag, y = acf)) +
  geom_bar(stat = "identity", width = 0.02, fill = "steelblue") +
  geom_hline(yintercept = 0, color = "black") +
  labs(title = "ACF Residuos", x = "Rezago", y = "ACF") +
  theme_minimal()

# ACF de residuos^2
acf_r2 <- acf(z$Residual^2, plot = FALSE)
acf_df2 <- with(acf_r2, data.frame(lag, acf))

p3 <- ggplot(acf_df2, aes(x = lag, y = acf)) +
  geom_bar(stat = "identity", width = 0.02, fill = "darkred") +
  geom_hline(yintercept = 0, color = "black") +
  labs(title = "ACF Residuos²", x = "Rezago", y = "ACF") +
  theme_minimal()

# QQ Plot

p4 <- ggplot(z, aes(sample = Residual)) +
  stat_qq() +
  stat_qq_line() +
  labs(title = "QQ-Plot Residuos", x = "Teórico", y = "Muestral") +
  theme_minimal()

grid.arrange(p1, p2, p3, p4, ncol = 2)

## Ljung-Box: no debería haber autocorrelación

Box.test(z$Residual,    lag = 20, type = "Ljung-Box")
Box.test(z$Residual^2,  lag = 20, type = "Ljung-Box")

ArchTest(z$Residual, lags = 10)   # p-value > 0.05 = sin ARCH residual importante

# Colas y distribucion 

hist(z$Residual, breaks = 40, main = "Residuos estandarizados", xlab = "")



# CREA UNA CARPETA PARA LOS ARCHIVOS DEL TABLERO
dir.create("datos_modelo", showWarnings = FALSE)

# 1. Retornos de todas las series (ya los tienes en df)
df_export <- df %>%
  as.data.frame() %>%
  tibble::rownames_to_column("Fecha")

df_export$Fecha <- as.Date(df_export$Fecha)

write.csv(df_export,
          file = "datos_modelo/retornos_usdcop.csv",
          row.names = FALSE)

# 2. Pronóstico vs Real (y_test, pred_mean)
pronostico_df <- data.frame(
  Fecha      = as.Date(rownames(X_test)),
  real       = y_test,
  pronostico = pred_mean
)

write.csv(pronostico_df,
          file = "datos_modelo/pronostico_usdcop.csv",
          row.names = FALSE)

# 3. Residuos estandarizados (z)
# z ya lo construiste en tu código:
# z <- residuals(fit_train, standardize = TRUE) %>% as.data.frame() %>%
#   rownames_to_column()
# colnames(z) <- c("Fecha", "Residual")
# z$Fecha <- as.Date(rownames(X_train)[-c(42:59)])

write.csv(z,
          file = "datos_modelo/residuos_usdcop.csv",
          row.names = FALSE)
