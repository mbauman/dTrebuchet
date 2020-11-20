using Trebuchet, Flux
using Zygote: forwarddiff
using Statistics: mean
using Random

# Intentionally destroy one of the stages:

const model = Chain(Dense(10, 16, Flux.leakyrelu),
                    Dense(16, 64, Flux.leakyrelu),
                    Dense(64, 16, Flux.leakyrelu),
                    Dense(16, 1)) |> f64
const ACTUAL = Ref(false)

function stage3!(du, u, pr::TrebuchetState, t)
    c = pr.c
    Grav = c.Grav
    ρ = c.ρ
    WS = c.w
    Aeff = π*(pr.l.z)^2
    mP = pr.m.p
    Cd = c.Cd

    Pvx = u[3]
    Pvy = u[4]

    dPx = Pvx
    dPy = Pvy
    dPvx = -(ρ*Cd*Aeff*(Pvx-WS)*sqrt(Pvy^2+(WS-Pvx)^2))/(2*mP)
    dPvy = -Grav - (ρ*Cd*Aeff*Pvy*sqrt(Pvy^2+(WS-Pvx)^2))/(2*mP)

    du[1] = dPx
    du[2] = dPy
    du[3] = dPvx
    du[4] = dPvy
end


shoot(ps) = forwarddiff(p -> shoot(p...), ps)
function shoot(wind, angle, weight)
    Trebuchet.shoot((wind, Trebuchet.deg2rad(angle), weight))[2]
end

function loss(wind, angle, weight)
    ACTUAL[] = false
    guess = shoot([wind, angle, weight])
    ACTUAL[] = true
    reality = shoot([wind, angle, weight])
    return (guess - reality)^2
end

lerp(x, lo, hi) = x*(hi-lo)+lo

DIST  = (20, 80)	# Angle bounds
WEIGHT = (100, 400) # Weight bounds
SPEED =   5 # Avg wind speed

randparams() = (randn() * SPEED, lerp(rand(), DIST...), lerp(rand(), WEIGHT...))

meanloss() = mean(sqrt(loss(randparams()...)) for i = 1:1000)

@time Flux.train!(loss, params(model), (randparams() for i in 1:100000), ADAM(), cb = Flux.throttle(()->@show(meanloss()), 100))
