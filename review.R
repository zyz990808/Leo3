IQR(c(1,2,3,4,5,6,7,8))
set.seed(123)
rnorm(n=10, 10, 3)
pnorm(5, 10, 3, lower.tail = FALSE)
pnorm(5, 10, 3, lower.tail = TRUE)
qnorm(0.9522096, 10, 3)
qnorm(0.04779035, 10, 3)
dnorm(10,10,3)
#syntax: dnorm(x, mean, sd)
log(4, base=2)
cos(sqrt(3))
sqrt(5)
x <- c(1,2,3,4,5,6,7,8)
summary(x)
table(x)
table(c(rep("Yellow", 20), rep("Green", 10), rep("Blue", 50)))
rep(1, 20)
table(as.factor(c(rep("Yellow", 20), rep("Green", 10), rep("Blue", 50))))
#Integrate
y <- function(x) x^2+x+1
integrate(y, -1, 2)
#derivatives
f <- expression(log(x)/log(a))
D(f, "x")
#Multiple Integrate
f <- function(x) {1}
adaptIntegrate(f, c(-1,-2), c(3,4))
#limit
x <- c(1:100)
y <- x*sin(1/x)
plot(x,y,type = "l")
x <- seq(0.99, 0.99999999999999999, by =0.00001)
y <- 1/(1-x)
plot(x,y,type = "l")
x <- c(1:100)
mean(x)
sum((x-mean(x))^2)
sum(x^2)
y <- c(101:200)
sum(x/y)
























