#%% Training
using Flux, Trebuchet
using Zygote: forwarddiff
using Statistics: mean
using Random

lerp(x, lo, hi) = x*(hi-lo)+lo

# Opt-in to forward mode AD
shoot(ps) = forwarddiff(p -> shoot(p...), ps)
function shoot(wind, angle, weight)
    Trebuchet.shoot((wind, Trebuchet.deg2rad(angle), weight))[2]
end

Random.seed!(0)

model = Chain(Dense(2, 16, σ),
              Dense(16, 64, σ),
              Dense(64, 16, σ),
              Dense(16, 2)) |> f64

θ = params(model)

function aim(wind, target)
  angle, weight = model([wind, target])
  angle = σ(angle)*90
  weight = weight + 200
  angle, weight
end

distance(wind, target) =
  shoot([wind, aim(wind, target)...])

function loss(wind, target)
    (distance(wind, target) - target)^2
end

DIST  = (20, 100) # target distance bounds
SPEED =   5 # Avg wind speed

randtarget() = (randn() * SPEED, lerp(rand(), DIST...))

dataset = (randtarget() for i = 1:10000)
ps = []
cb = Flux.throttle(1) do
    p = plot(x->shoot(5, aim(5, x)...)-x, 20, 100, label="", ylabel="error (m)", xlabel="distance (m)", ylims=(-10,10))
    display(p)
    push!(ps, p)
end

@time Flux.train!(loss, θ, dataset, ADAM(), cb = cb)

#%% 

anim = @animate for p in ps
    plot(p)
end
gif(anim)