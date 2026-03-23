To extract data for the infinitesimal kappa (kappa --> 0) we use the script named "infinitesimal_kappa_new.jl". The command is given by

>>> julia   infinitesimal_kappa_new.jl    *LLL*   *Gamma*  *hhh*    *runseed*   *disorder*

*LLL*: System size
*Gamma*: Cavity_chain coupling
*hhh*: transverse field
*runseed*: disorder seeding (authomaticaly create seed for different disorders)
*disorder*: disorder realization tag (for example, 1st, 10th, 256th and so on)

Output:
-- lT: List of local temperature
-- A = make_A(U,job): Transition Matrix

--------------------------------------------------------------------------------
To extract data for the finite kappa (kappa >> 0) we use the script named "finite_kappa.jl". The command is given by

>>> julia  finite_kappa.jl

Output:
-- lT: List of local temperature
-- A = make_A(U,job): Transition Matrix

* In both cases, you can change the code manually, so the ARGS fit the user preferences.
_______________________________________________________________________________
Sample of cluster run (infinitesimal kappa (kappa --> 0))
The script "run_231121_smallkappa_KokiCode.template" create a b## folder for each parameter in the *_builder file, and th lines

>>>  for i in {1..*ItNum*}
>>>  do
>>>      { time  julia ./$EXEC1  *LLL*   *Gamma*  *hhh*    *runseed*   $i  ; } > temp1.out 2> error1.err 
>>>  done

rerun the *.jl script for each disorder realization tag
