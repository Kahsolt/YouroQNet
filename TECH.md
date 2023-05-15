# Technical & Theoretical

    Theory is grey, evergreen the tree of life!

----

### Simple facts

⚪ rotation gates

```
[numerical semantics]
let |phi> = α|0> + β|1>
  RX transforms the amplitude for |0>/|1> (balancing modulo between α and β)
  RZ transforms the relative phase of α/β (balancing real/imag part within α and β, keeping modulo unchanged)
  RY transforms both the amplitude and relative phase, but dependently to each other

[rotation cycle on basis (period: 4*pi)]
  Θ    -3*pi -2*pi  -pi    0    pi    2*pi   3*pi
RX|0>  -i|1>  -|0>  i|1>  |0>  -i|1>  -|0>   i|1>   ; mult by -i then flip
RY|0>    |1>  -|0>  -|1>  |0>    |1>  -|0>   -|1>   ; flip, sign-toggle in period 2*pi
RZ|0>  -i|0>  -|0>  i|0>  |0>  -i|0>  -|0>   i|0>   ; mult by -i

=> this gives the input value range for QNN outcome with pmeasure
  is [-pi, pi], because the observable perioid is 2*pi (e.g. from -|b> to |b>)

[compose/decompose law]
  R(x + y) = R(x) * R(y), where R is from [RX, RY, RZ]
hence we have:
  R(x + (-x)) = R(x) * R(-x) = I = R(0)
  R^(-1)(x) = R(-x)                  // let R^(-1) be inverse of R
  R(k*x) = Πk R(x) = R^k(x), k ∈ N
  R(w*x) = R^w(x), w ∈ R            // relax k to real number

[reduction on inner-linear-conposition]
  R(f(x)), where f(x) = w*x+b is a linear function
  = R(w*x + b)
  = R^w(x) * R(b)
```

----

by Armit
2023/05/14 
