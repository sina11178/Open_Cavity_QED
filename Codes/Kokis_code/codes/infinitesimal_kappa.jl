#Perturbation theory for kappa（b-boson representation）
using LinearAlgebra
using Plots
using Random
using BenchmarkTools
using Dates
using Statistics
using NPZ


struct Job #parameters
    J::Float64 #Ising
    Gamma::Float64 #coupling
    h::Float64#Transverse Field
    mu::Float64 #disorder

    omega::Float64 #frequency
    beta::Float64 #pumping
    kappa::Float64 #dissipation
    omegaD::Float64 #bath energy

    Lspin::Int64 #spin number
    Pnum::Int64 #photon number cutoff
end

J = -1.07
Lspin = parse(Int64,ARGS[1])
Gamma = parse(Float64,ARGS[2])
h = parse(Float64,ARGS[3])
runseed =  10 #parse(Int64,ARGS[4])
disorder = 0 #parse(Int64,ARGS[5])
#Gamma = 0.5
mu = 0.2
omega = pi/0.8
beta = 4.0
#kappa = parse(Float64,ARGS[2])
kappa = 0.0
omegaD = 10.0

#Lspin = parse(Int64,ARGS[3])
#Pnum = parse(Int64,ARGS[4])
Pnum = 10

if isfile("Temperature_L"*string(Lspin)*"_Gamma"*string((round(Gamma, digits=4)))*"_seed"*string(runseed+disorder)*".npy")
    println("file already exists")
    exit()
    end


job = Job(J,Gamma,h,mu,omega,beta,kappa,omegaD,Lspin,Pnum)


function make_muj(job)
    #seed = Dates.value(now())+runseed #現在時刻取得
    #seed = Dates.value(now()) + 1000000000*parse(Int64,ARGS[5])
    seed = runseed+disorder
    println("seed = $seed")
    rng = MersenneTwister(seed)
    temp = rand(rng,Float64,job.Lspin) #uniform distribution in [0,1)
    id = ones(Float64,job.Lspin) #id=[1,1,1,1,,,]

    muj = 2*job.mu .*temp - job.mu .*id
    println("muj = ",muj)

    return muj
end

function make_Hamiltonian(muj, job) #dense matrix
    D = (2^job.Lspin) * job.Pnum
    DL = 2^job.Lspin

    Ham = zeros(Float64,D,D)

    #対角成分
    for a = 0:D-1
        x = 0

        #sigmaz*simgaz
        for j = 0:job.Lspin-1
            if j == job.Lspin - 1
                j1 = job.Lspin - 1
                j2 = 0
            else
                j1 = j
                j2 = j + 1
            end

            s1 = div(a, 2^j1) % 2
            s2 = div(a, 2^j2) % 2

            if s1 == 0 && s2 == 0
                x += job.J
            elseif s1 == 1 && s2 == 0
                x -= job.J
            elseif s1 == 0 && s2 == 1
                x -= job.J
            else
                x += job.J
            end
        end

        #sigmaz
        for j = 0:job.Lspin-1
            s = div(a, 2^j) % 2

            if s == 0
                x += -muj[j+1] #list starts from 1
            else
                x += muj[j+1]
            end
        end

        #omega b^dag b
        x += job.omega * div(a, 2^job.Lspin)

        #insert
        m = a + 1
        Ham[m,m] += x

    end

    #sigmax
    coef = -(8*job.beta*job.omega*job.Gamma / (job.kappa^2 + 4*job.omega^2))-job.h
    for a = 0:D-1
        for j = 0:job.Lspin-1
            s,b = flip(a,j)
            m = b+1
            n = a+1
            Ham[m,n] += coef
        end
    end



    #sigmax b
    for a = 0:D-1
        num = div(a,DL)
        if num != 0
            for j = 0:job.Lspin-1
                s,b = flip(a,j)
                m = b+1 - DL
                n = a+1
                Ham[m,n] += job.Gamma * sqrt(num)
            end
        end
    end

    #sigmax b^dag
    for a = 0:D-1
        num = div(a,DL)
        if num != job.Pnum-1
            for j = 0:job.Lspin-1
                s,b = flip(a,j)
                m = b+1 + DL
                n = a+1
                Ham[m,n] += job.Gamma * sqrt(num+1)
            end
        end
    end

    return Ham
end


function flip(b,j) #j=0,1,2,,,
    s = div(b,2^j)%2
    if s==0
        c = b + 2^j
    else
        c = b - 2^j
    end

    return s,c
end

function make_A(U, job)
    D = (2^job.Lspin) * job.Pnum
    DL = 2^job.Lspin

    A = zeros(Float64, D, D)

    #m<=n
    temp = zeros(ComplexF64, D)
    for n = 1:D

        #先にa|n>を計算しとく
        temp = zeros(ComplexF64, D)
        for a = 0:D-1
            num = div(a, DL)
            if num != 0
                b = a - DL
                temp[b+1] = sqrt(num) * U[a+1, n]
            end
        end

        for m = 1:D
            x::ComplexF64 = 0.0
            for a = 1:D
                x += conj(U[a, m]) * temp[a]
            end
            A[m, n] = (abs(x))^2

        end
    end


    #m=n
    for m = 1:D
        x::ComplexF64 = 0.0
        for a = 0:D-1
            num = div(a, DL)
            if num != 0
                x += conj(U[a+1, m]) * num * U[a+1, m]
            end
        end
        A[m, m] -= x

    end

    return A
end

function search_ss(evA,UA,job)
    D = (2^job.Lspin) * job.Pnum
    thres = 1.0e-14  # NOTE: Sometimes, it might be on order of 1e-13 instead of 1e-14

    ss::Int64 = 0
    for a=1:D
        if abs(evA[a]) < thres
            ss = a
            break
        end
    end

    if ss==0
       println("Error!!!!!!!!No steady state")
    end

    rhoss_temp = UA[:,ss]

    #normalize
    sum = mean(rhoss_temp) * size(rhoss_temp,1)
    rhoss_temp ./= sum

    rhoss = real(rhoss_temp)

    return rhoss
end





#thermal fluctuation
function cal_localT(j,rhoss,ev,U,job) #j=0,1,2,3,,,

    sigmajy2 = make_sigmajy2(j,U,job)

    ite = 15
    Tem = zeros(Float64,ite)
    ecur = zeros(Float64,ite)
    slope = cal_slope(rhoss,ev,sigmajy2,job)

    Tem[1] = 500.0
    ecur[1] = cal_ecur(rhoss,ev,sigmajy2,Tem[1],job)
    Tem[2] = Tem[1] - ecur[1]/slope
    #println("Slope: ", slope)
    #println("j=", j, "sig_jy_squared min=", minimum(sigmajy2), "max=", maximum(sigmajy2))

    for i=2:ite-1
        ecur[i] = cal_ecur(rhoss,ev,sigmajy2,Tem[i],job)
        if abs(ecur[i] - ecur[i-1]) < 1.0e-12 && abs(ecur[i]) < 1.0e-12
            Tem[ite] = Tem[i]
            break
        end
        Tem[i+1] = (ecur[i-1]*Tem[i] - Tem[i-1]*ecur[i])/(ecur[i-1]-ecur[i])
        #println("Tem = ",Tem[i],", ecur = ",ecur[i])
    end

    return Tem[ite]
end

function make_sigmajy2(j,U,job) #j=0,1,2,,,
    D = (2^job.Lspin) * job.Pnum

    sigmajy2 = zeros(Float64,D,D)

    #m>=n
    temp = zeros(ComplexF64,D)
    for n=1:D

        #先にsigmajy |n>を計算しとく
        temp = zeros(ComplexF64,D)
        for a=0:D-1
            s,b = flip(a,j)
            if s==0
                temp[b+1] += -im * U[a+1,n]
            else
                temp[b+1] += +im * U[a+1,n]
            end
        end

        for m=n:D

            x::ComplexF64 = 0.0
            for a=1:D
                x += conj(U[a,m]) * temp[a]
            end
            sigmajy2[m,n] += (abs(x))^2

        end
    end

    #m<n
    for m=1:D
        for n=m+1:D
            sigmajy2[m,n] = sigmajy2[n,m]
        end
    end

    return sigmajy2
end

function cal_ecur(rhoss,ev,sigmajy2,Tem,job)
    D = (2^job.Lspin) * job.Pnum

    ecur = 0.0
    for m=1:D
        for n=1:D

            if m != n
                dE = ev[m] - ev[n]
                DB = dE/(dE^4 + job.omegaD^4)
                if abs(dE/Tem) < 1.0e-6
                    nB = Tem/dE
                else
                    nB = 1.0/(exp(dE/Tem)-1.0)
                end
                ecur += dE * sigmajy2[m,n] * rhoss[n] * nB * DB
            end

        end
    end

    return ecur
end

function cal_slope(rhoss,ev,sigmajy2,job)
    D = (2^job.Lspin) * job.Pnum

    slope = 0.0
    for m=1:D
        for n=1:D
            dE = ev[m] - ev[n]
            DB = dE/(dE^4 + job.omegaD^4)

            slope += sigmajy2[m,n] * rhoss[n] * DB

        end
    end

    return slope
end









############################################ Main
#function mainf(job)
D = (2^job.Lspin) * job.Pnum
D2 = D^2

#disorder
muj = make_muj(job)


#Hamiltonian
Ham = make_Hamiltonian(muj,job)
ev,U = eigen(Ham)


#transfer matrix
A = make_A(U,job)
evA,UA = eigen(A)
#println(evA)
#plot(real(evA),imag(evA),st=scatter)



#search and normalize the steady state
rhoss = search_ss(evA,UA,job)
#println(rhoss)



#check
# sigmajy2 = make_sigmajy2(3,U,job)
# Tem = Vector(0.1:0.1:2)
# ecur = zeros(Float64,20)
# for i=1:20
#     ecur[i] = cal_ecur(rhoss,ev,sigmajy2,Tem[i],job)
# end




#local temperatures
lT = zeros(Float64,job.Lspin)
for j=1:job.Lspin
    lT[j] = cal_localT(j-1,rhoss,ev,U,job)
end


# check whether negative temperature appears or not
for i=1:job.Lspin
    if lT[i]<0
        println("Negative temperature!")
    end
end

#mean and standard deviation of temperature fluctuation
println("lT: ", lT)
Tbar = mean(lT)
#npzwrite("Temperature_L"*string(Lspin)*"_Gamma"*string((round(Gamma, digits=4)))*"_seed"*string(runseed+disorder)*".npy", lT)
delT = std(lT;corrected=false)

println(lT)
println(Tbar, " ", delT)
println(delT/Tbar)



#plot(Tem,ecur)
#end




# for i=1:500
#     mainf(job)
# end
