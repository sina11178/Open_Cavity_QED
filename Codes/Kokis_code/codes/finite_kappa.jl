using LinearAlgebra
using Arpack
using SparseArrays
using Random
using BenchmarkTools
using Dates
using Statistics
using IterativeSolvers
using KrylovKit

struct Job #parameters
    J::Float64 #ising interaction
    Gamma::Float64 #coupling
    mu::Float64 #random field

    omega::Float64 #frequency
    beta::Float64 #pumping
    kappa::Float64 #dissipation
    omegaD::Float64 #実数

    Lspin::Int64 #spin number
    Pnum::Int64 #photon number cutoff
end

J = -1.07

#Gamma = parse(Float64,ARGS[1])
#GammaL = parse(Float64,ARGS[1])
Gamma = 0.2
mu = 1.3
omega = pi/0.8
beta = 4.0
#kappa = parse(Float64,ARGS[2])
kappa = 1.0
omegaD = 10.0

#Lspin = parse(Int64,ARGS[3])
#Pnum = parse(Int64,ARGS[4])
Lspin = 4
Pnum = 10
#Gamma = GammaL * Lspin^(-1.0/6.0)

job = Job(J,Gamma,mu,omega,beta,kappa,omegaD,Lspin,Pnum)





function make_sigmajy(j,job) #j=1,2,,,L
    size = (2^job.Lspin) * job.Pnum
    size2 = size*size
    row = zeros(Int64,size2)
    col = zeros(Int64,size2)
    val = zeros(ComplexF64,size2)

    for a=0:size-1
        for b=0:size-1
            s = div(b,2^(j-1))%2
            if s==0
                c = b + 2^(j-1)
                m = size*a + b + 1
                n = size*a + c + 1
                row[m] = n
                col[m] = m
                val[m] = -im
            else
                c = b - 2^(j-1)
                m = size*a + b + 1
                n = size*a + c + 1
                row[m] = n
                col[m] = m
                val[m] = +im
            end
        end
    end

    sigmajy = sparse(row,col,val)
    return sigmajy
end

function make_Liouvillian(muj,job) #j=0,1,2,,,L-1
    size = (2^job.Lspin) * job.Pnum

    row = Int64[]
    col = Int64[]
    val = ComplexF64[]

    ##################################
    ###-i (I * H)
    #対角
    for a=0:size-1
        for b=0:size-1
            x = 0

            #sigmaz*simgaz
            for j=0:job.Lspin-1
                if j==job.Lspin-1
                    j1=job.Lspin-1; j2=0
                else
                    j1=j; j2=j+1
                end

                s1 = div(b,2^j1)%2
                s2 = div(b,2^j2)%2

                if s1==0 && s2==0
                    x += job.J
                elseif s1==1 && s2==0
                    x -= job.J
                elseif s1==0 && s2==1
                    x -= job.J
                else
                    x += job.J
                end
            end

            #sigmaz
            for j=0:job.Lspin-1
                s = div(b,2^j)%2

                if s==0
                    x += -muj[j+1] #list starts from 1
                else
                    x += muj[j+1]
                end
            end

            #omega b^dag b
            x += job.omega * div(b,2^job.Lspin)

            #insert
            m = size*a+b+1
            push!(row,m)
            push!(col,m)
            push!(val,-im * x)

        end
    end


    #sigmax
    cof = -8 * job.beta * job.omega * job.Gamma / (job.kappa^2 + 4*job.omega^2)
    for a=0:size-1
        for b=0:size-1
            for j=0:job.Lspin-1
                s,c = flip(b,j)

                #insert
                m = size*a+c+1
                n = size*a+b+1
                push!(row,m)
                push!(col,n)
                push!(val,-im * cof)
            end
        end
    end

    #sigmax b
    for a=0:size-1
        for b=0:size-1
            for j=0:job.Lspin-1
                num = div(b,2^job.Lspin) #photon number of b-state
                s,c = flip(b,j)

                if num != 0
                    c -= 2^job.Lspin

                    #insert
                    m = size*a+c+1
                    n = size*a+b+1
                    push!(row,m)
                    push!(col,n)
                    push!(val,-im * job.Gamma * sqrt(num))
                end

            end
        end
    end

    #sigmax b^dag
    for a=0:size-1
        for b=0:size-1
            for j=0:job.Lspin-1
                num = div(b,2^job.Lspin) #photon number of b-state
                s,c = flip(b,j)

                if num != job.Pnum-1
                    c += 2^job.Lspin

                    #insert
                    m = size*a+c+1
                    n = size*a+b+1
                    push!(row,m)
                    push!(col,n)
                    push!(val,-im * job.Gamma * sqrt(num+1))
                end

            end
        end
    end

    ##################################
    ###i (H^T * I)
    #対角
    for a=0:size-1
        for b=0:size-1
            x = 0

            #sigmaz*simgaz
            for j=0:job.Lspin-1
                if j==job.Lspin-1
                    j1=job.Lspin-1; j2=0
                else
                    j1=j; j2=j+1
                end

                s1 = div(b,2^j1)%2
                s2 = div(b,2^j2)%2

                if s1==0 && s2==0
                    x += job.J
                elseif s1==1 && s2==0
                    x -= job.J
                elseif s1==0 && s2==1
                    x -= job.J
                else
                    x += job.J
                end
            end

            #sigmaz
            for j=0:job.Lspin-1
                s = div(b,2^j)%2

                if s==0
                    x += -muj[j+1]
                else
                    x += muj[j+1]
                end
            end

            #omega b^dag b
            x += job.omega * div(b,2^job.Lspin)

            #insert
            m = size*b+a+1
            push!(row,m)
            push!(col,m)
            push!(val,im * x)

        end
    end


    #sigmax
    cof = -8 * job.beta * job.omega * job.Gamma / (job.kappa^2 + 4*job.omega^2)
    for a=0:size-1
        for b=0:size-1
            for j=0:job.Lspin-1
                s,c = flip(b,j)

                #insert
                m = size*c+a+1
                n = size*b+a+1
                push!(row,n)
                push!(col,m)
                push!(val,im * cof)
            end
        end
    end

    #sigmax b
    for a=0:size-1
        for b=0:size-1
            for j=0:job.Lspin-1
                num = div(b,2^job.Lspin) #photon number of b-state
                s,c = flip(b,j)

                if num != 0
                    c -= 2^job.Lspin

                    #insert
                    m = size*c+a+1
                    n = size*b+a+1
                    push!(row,n)
                    push!(col,m)
                    push!(val,im * job.Gamma * sqrt(num))
                end

            end
        end
    end

    #sigmax b^dag
    for a=0:size-1
        for b=0:size-1
            for j=0:job.Lspin-1
                num = div(b,2^job.Lspin) #photon number of b-state
                s,c = flip(b,j)

                if num != job.Pnum-1
                    c += 2^job.Lspin

                    #insert
                    m = size*c+a+1
                    n = size*b+a+1
                    push!(row,n)
                    push!(col,m)
                    push!(val,im * job.Gamma * sqrt(num+1))
                end

            end
        end
    end


    ##################################
    #b^star * b
    for a=0:size-1
        for b=0:size-1
            num1 = div(a,2^job.Lspin)
            num2 = div(b,2^job.Lspin)

            if num1 != 0 && num2 != 0
                c = a - 2^job.Lspin
                d = b - 2^job.Lspin

                #insert
                m = size*c+d+1
                n = size*a+b+1
                push!(row,m)
                push!(col,n)
                push!(val,job.kappa * sqrt(num1) * sqrt(num2))
            end

        end
    end

    #(-1/2) I * a^dag a
    for a=0:size-1
        for b=0:size-1
            num = div(b,2^job.Lspin)

            if num != 0
                #insert
                m = size*a+b+1
                push!(row,m)
                push!(col,m)
                push!(val,-job.kappa * num/2)
            end

        end
    end

    #(-1/2) (a^dag a)^T * I
    for a=0:size-1
        for b=0:size-1
            num = div(b,2^job.Lspin)

            if num != 0
                #insert
                m = size*b+a+1
                push!(row,m)
                push!(col,m)
                push!(val,-job.kappa * num/2)
            end

        end
    end

    L = sparse(row,col,val)
    return L
end

function flip(b,j) #j=0,1,2,,,,L-1
    s = div(b,2^j)%2
    if s==0
        c = b + 2^j
    else
        c = b - 2^j
    end

    return s,c
end

function make_muj(job)
    #seed = Dates.value(now()) + 1000000000*parse(Int64,ARGS[5])  #現在時刻取得
    #seed = Dates.value(now()) + 1000000000
    seed = 1
    rng = MersenneTwister( seed )
    temp = rand(rng,Float64,job.Lspin) #uniform distribution in [0,1)
    id = ones(Float64,job.Lspin) #id=[1,1,1,1,,,]

    muj = 2*job.mu .*temp - job.mu .*id
    #println("muj = ",muj)

    return muj
end

function normalize_sm(vec,job)
    size = (2^job.Lspin) * job.Pnum
    tr = 0
    for a=0:size-1
        m = size*a+a+1
        tr += vec[m]
    end

    vec ./= tr
    return 0
end

function tr_sm(vec,job)
    size = (2^job.Lspin) * job.Pnum
    tr = 0.0
    for a=0:size-1
        m = size*a+a+1
        tr += vec[m]
    end

    return tr
end



############################################## Energy current
#efficient calculation
function cal_localT_eff(rho,j,L,job) #j=1,2,3,,,,
    wDp = job.omegaD * (1+im)/sqrt(2.0)
    wDm = job.omegaD * (1-im)/sqrt(2.0)

    sigmajy = make_sigmajy(j,job)

    trp = cal_trsLsr(rho,L,wDp,sigmajy,job)
    trm = cal_trsLsr(rho,L,wDm,sigmajy,job)
    slope = imag( trp/wDm^2 + trm/wDp^2 )/4.0
    #println("slope = ",slope)

    #iteration
    ite = 10
    Tem = zeros(Float64,ite)
    ecur = zeros(Float64,ite)

    Tem[1] = 10.0
    ecur[1] = cal_ecur_eff(rho,L,sigmajy,trp,trm,Tem[1],job)
    Tem[2] = Tem[1] - ecur[1]/slope

    for i=2:ite-1
        ecur[i] = cal_ecur_eff(rho,L,sigmajy,trp,trm,Tem[i],job)
        Tem[i+1] = (ecur[i-1]*Tem[i] - Tem[i-1]*ecur[i])/(ecur[i-1]-ecur[i])
        #println("Tem = ",Tem[i],", ecur = ",ecur[i])
    end

    return Tem[ite]
end

function cal_trsLsr(rho,L,shift,sigmajy,job)
    D = (2^job.Lspin) * job.Pnum
    D2 = D^2

    id = sparse(I,D2,D2)

    Lshift = L - shift .* id
    dm1 = sigmajy * rho
    #dm2 = Lshift \ dm1
    dm2 = bicgstabl(Lshift,dm1)
    dm3 = sigmajy * dm2
    trsLsr = tr_sm(dm3,job)

    return trsLsr
end

function cal_ecur_eff(rho,L,sigmajy,trp,trm,Tem,job)
    D = (2^job.Lspin) * job.Pnum
    D2 = D^2

    id = sparse(I,D2,D2)
    dm1 = sigmajy * rho
    ecur = 0.0

    #1&2
    wDp = job.omegaD * (1+im)/sqrt(2.0)
    wDm = job.omegaD * (1-im)/sqrt(2.0)
    cp = -1.0/(4.0 * wDm * (exp(-wDm/Tem)-1))
    cm = 1.0/(4.0 * wDp * (exp(wDp/Tem)-1))

    ecur += imag(cp*trp + cm*trm)

    #3
    nmax = floor(Int,5*job.omegaD/Tem) + 10

    for n=1:nmax
        coef = - (2*n*pi)^2 * Tem^3 / (job.omegaD^4 + (2*n*pi*Tem)^4);
        shift = 2*n*pi*Tem;

        Lshift = L - shift .* id
        #dm2 = Lshift \ dm1
        dm2 = bicgstabl(Lshift,dm1)
        dm3 = sigmajy * dm2
        ecur += imag(coef * tr_sm(dm3,job))
    end

    return ecur
end





#steady state by time evoulution
function ss_by_dyn(muj,L,job)
    D = (2^job.Lspin) * job.Pnum
    D2 = D^2

    t = 0.0
    dt = 1.0
    Nstep = 2000
    tmax = 1000 * (0.3/job.Gamma)

    #initial state
    rho = zeros(ComplexF64,D2)
    rho[1] = 1.0


    dif1::Float64 = 0.0
    dif2::Float64 = 0.0

    #time evolution
    for i=1:Nstep
        rho1,info = exponentiate(L,dt,rho)

        #convergence check 3
        if i==100
            dif1 = cal_dif(rho,rho1)
        end

        if i%10==0 && i>100
            dif2 = cal_dif(rho,rho1)
            Nrho = norm(rho)

            y = dif1/dif2
            if y<1
                #println("Error : y<1")
                dif1 = dif2
            else
                c = log(y)/10.0
                x = abs(dif2/(c*Nrho))

                dif1 = dif2

		#break point
                if (x<1e-12) || (x<1e-8 && t>tmax)
                    break
                end
            end

        end

        #=
        if i==Nstep
            println("Not converged! Final x = $x")
            #output to file
            Gc = sprintf1("%.2f",job.Gamma)
            kc = sprintf1("%.1f",job.kappa)
            Lc = string(job.Lspin)
            Nc = string(job.Pnum)

            fp = open("notconv_G$(Gc)_k$(kc)_L$(Lc)_N$(Nc).txt","a")
            println(fp," ")
            for j=1:job.Lspin
                print(fp,muj[j]," ")
            end
            close(fp)
        end
        =#


        rho = rho1
        t += dt
    end

    return rho
end




function cal_dif(rho,rho1)
    drho = rho - rho1
    x = norm(drho)

    return x
end


############################################ Main
function f(job)
D = (2^job.Lspin) * job.Pnum
D2 = D^2

#disorder
muj = make_muj(job)

#Liouvillian
L = make_Liouvillian(muj,job)

#Below, we can use one of two ways to calculate the steady state.
#The first is using eigs-function.
#The second is calculating the time-evolution by krylov method for a long time.
#Basically, the first method is more reliable, but the second would be faster for a large system.

#steady state
val,vec = eigs(L,nev=1,which=:SM)
rho = vec[:,1]
normalize_sm(rho,job)

#steady state 2
#rho = ss_by_dyn(muj,L,job)



#local temperatures
lT = zeros(Float64,job.Lspin)
for j=1:job.Lspin
    lT[j] = cal_localT_eff(rho,j,L,job)
    #println("#lT finished")
end

#mean and standard deviation
Tbar = mean(lT)
delT = std(lT;corrected=false) #falseにすると/sqrt(n)、trueだと/sqrt(n-1)になる

println("test")
println(Tbar," ",delT)

return 0
end


for i=1:1
f(job)
end
