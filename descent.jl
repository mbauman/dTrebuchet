using Flux, Trebuchet
using Zygote: forwarddiff
using Statistics: mean
using Random

# Opt-in to forward mode AD
shoot(ps) = forwarddiff(p -> shoot(p...), ps)
function shoot(wind, angle, weight)
    Trebuchet.shoot((wind, Trebuchet.deg2rad(angle), weight))[2]
end

# Gradient descent to a single target:
function target!(f, args; N = 10, η = 0.1)
    for _ in 1:N
        grads = f'(args)
        args .-= η .* grads
    end
    return args
end

target!(x->(shoot(5, x...)-100)^2, [55.0, 220.0])