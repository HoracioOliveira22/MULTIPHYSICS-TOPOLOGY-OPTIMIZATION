using Pkg
Pkg.activate()
import Pkg
# Add here your .msh file with all the corresponding labels, file must be created with Gmsh or GridapGmsh.jl
mesh_file_path = "Your/path/to/.msh" 
# Setup Environment, specially if used for the first time 
function setup_environment()
    Pkg.activate(".")
    if Sys.iswindows()
        try
            installed = haskey(Pkg.project().dependencies, "P4est_jll")
            if !installed
                Pkg.add(name="P4est_jll", version="2.8.1") #Further versions might cause issues in windows 
                Pkg.pin("P4est_jll")
            else
                println("P4est_jll is already installed.")
            end
        catch e
            @warn "Failed to pin P4est_jll. This might cause GridapTopOpt precompilation errors on Windows." exception=e
        end
    end
    required_packages = [
        "LineSearches",
        "SparseMatricesCSR",
        "WriteVTK",
        "Gridap",
        "PartitionedArrays",
        "GridapDistributed",
        "GridapPETSc",
        "Metis",
        "GridapGmsh",
        "NLopt",
        "GridapTopOpt",
        "ChainRulesCore",
        "GridapSolvers"
    ]
    for pkg in required_packages
        if Base.find_package(pkg) === nothing
            println("  -> Installing missing package: $pkg...")
            Pkg.add(pkg)
        end
    end
    Pkg.instantiate()
    println("Environment ready.")
end

setup_environment()

using LineSearches
using LinearAlgebra
using SparseArrays
using SparseMatricesCSR
using WriteVTK
using Gridap
using Gridap.Algebra
using Gridap.CellData
using Gridap.FESpaces
using Gridap.Fields
using Gridap.Geometry
using Gridap.Helpers
using Gridap.ReferenceFEs
using Gridap.MultiField
import Gridap.Geometry: get_node_coordinates
import Gridap.FESpaces: SparseMatrixAssembler, AssemblyStrategy, DefaultAssemblyStrategy, SparseMatrixBuilder, ArrayBuilder
import Gridap.MultiField: BlockMultiFieldStyle, MultiFieldStyle, BlockSparseMatrixAssembler
using PartitionedArrays
using GridapDistributed
using GridapPETSc
using GridapPETSc.PETSC
using Metis
using GridapGmsh
using NLopt
import NLopt: optimize!
using GridapTopOpt
using ChainRulesCore
using GridapSolvers
using GridapSolvers.LinearSolvers
using GridapSolvers.MultilevelTools
using GridapSolvers.NonlinearSolvers


import GridapTopOpt: AbstractFEStateMap
import GridapTopOpt: IntegrandWithMeasure, StateParamIntegrandWithMeasure
import GridapTopOpt: get_measure, get_trial_space, get_state, get_spaces, get_assemblers, get_deriv_space, get_deriv_assembler
import GridapTopOpt: forward_solve!, update_adjoint_caches!, adjoint_solve!, dRdφ
import Base: nameof

# MultiField Assembly
function Gridap.FESpaces.SparseMatrixAssembler(
    mat, vec,
    trial::MultiFieldFESpace{MS},
    test ::MultiFieldFESpace{MS},
    strategy::AssemblyStrategy=DefaultAssemblyStrategy()
) where MS <: BlockMultiFieldStyle
    mfs = MultiFieldStyle(test)
    return BlockSparseMatrixAssembler(mfs,trial,test,SparseMatrixBuilder(mat),ArrayBuilder(vec),strategy)
end

# HELPERS

function get_local_values!(v::PVector,w::Vector{Float64})
  map(local_values(v),v.index_partition) do v,part
    gids = local_to_global(part)
    v[:] = w[gids]
  end
end

function copy_local_values!(w::Vector{Float64}, v, nd::Int64)
  map(v) do vl
      if hasproperty(vl, :data)
          source = vl.data
      else
          source = vl
      end
      
      n = length(source)
      if n > 0
          w[1:n] = source
      end
  end
end

function get_cell_size_squared(Ω::GridapDistributed.DistributedTriangulation,dΩ)
  D = num_cell_dims(Ω)
  a = map(local_views(∫(1)dΩ)) do d
    collect(get_array(d)).^(2/D)
  end
  return CellField(a,Ω)
end

function set_design_domain(is_design,model)
  if is_design === :all
    labels = get_face_labeling(model)
    add_tag_from_tags!(labels,"design","interior")
    add_empty_tag!(labels,"nodesign")
  end
  return nothing
end

function add_empty_tag!(labels::GridapDistributed.DistributedFaceLabeling,tag)
  map(local_views(labels)) do l
    new=num_entities(l)+1
    add_tag!(l,tag,[new])
  end
end

# SOLVERS

const PETScTypes = Union{Val{:petsc},Val{:petsc_mumps},Val{:petsc_amg},Val{:petsc_cg_jacobi},Val{:petsc_cg_amg},Val{:petsc_elasticity}}

function default_solver_params(::Val{:julia})
    return Dict(
      :matrix_type    => SparseMatrixCSC{Float64,Int64},
      :vector_type    => Vector{Float64},
      :petsc_options  => "",
      :niter          => 1,
      :rtol           => 1e-5,
    )
end

function default_solver_params(solver::Symbol)
   d = default_solver_params(Val(solver))
   d[:solver] = solver
   return d
end

@inline uses_petsc(solver::Dict) = uses_petsc(solver[:solver])
@inline uses_petsc(solver::Symbol) = uses_petsc(Val(solver))
uses_petsc(::Val{:julia}) = false
uses_petsc(::A) where A<:PETScTypes = true 

@inline get_solver(p::Dict,V) = get_solver(p[:solver],V)
@inline get_solver(s::Symbol,V) = get_solver(Val(s),V)
get_solver(::Val{:julia},V) = LUSolver()
get_solver(::Val{:petsc},V) = PETScLinearSolver()

function run_with_backend(f::Function,solvers::Vector;backend=:sequential,np=(1,1),kwargs...)
  solver_is_petsc = map(uses_petsc,solvers)
  use_petsc = maximum(solver_is_petsc)
  
  if use_petsc
    solver = solvers[findfirst(solver_is_petsc)]
    if isa(solver, Symbol)
      params = default_solver_params(solver)
    elseif isa(solver, Dict)
      params = solver
    end
  end

  if backend === :sequential
    with_debug() do distribute
      ranks = distribute(LinearIndices((prod(np),)))
      if use_petsc
        GridapPETSc.with(args=split(params[:petsc_options])) do
          f(ranks,np,solvers;kwargs...)
        end
      else
        f(ranks,np,solvers;kwargs...)
      end
    end
  elseif backend === :mpi 
    with_mpi() do distribute
      ranks = distribute(LinearIndices((prod(np),)))
      if use_petsc
        GridapPETSc.with(args=split(params[:petsc_options])) do
          f(ranks,np,solvers;kwargs...)
        end
      else
        f(ranks,np,solvers;kwargs...)
      end
    end
  else
    @error "Backend not available"
  end
end

# GRIDAPTOPOPT EXTENSIONS

Base.nameof(I::StateParamIntegrandWithMeasure) = Base.nameof(I.F)
Base.nameof(I::IntegrandWithMeasure) = Base.nameof(I.F)

# SelfAdjointAffineFEStateMap

struct SelfAdjointAffineFEStateMap{A,B,C,D,E,F} <: AbstractFEStateMap
  biform     :: A
  liform     :: B
  spaces     :: C
  param      :: D
  plb_caches :: E
  pde_caches :: F
  function SelfAdjointAffineFEStateMap(
    a::Function,l::Function,
    U,V,V_φ,U_reg,φh,dΩ...;
    assem_U = SparseMatrixAssembler(U,V),
    assem_deriv = SparseMatrixAssembler(U_reg,U_reg),
    ls::LinearSolver = LUSolver()
  )
    biform = IntegrandWithMeasure(a,dΩ)
    liform = IntegrandWithMeasure(l,dΩ)
    spaces = (U,V,V_φ,U_reg)
    param = copy(get_free_dof_values(φh))

    ## Pullback cache
    uhd = zero(U)
    vecdata = collect_cell_vector(U_reg,∇(biform,[uhd,uhd,φh],3))
    dudφ_vec = allocate_vector(assem_deriv,vecdata)
    plb_caches = (dudφ_vec,assem_deriv)

    ## Forward cache
    op = AffineFEOperator((u,v)->biform(u,v,φh),v->liform(v,φh),U,V,assem_U)
    K, b = get_matrix(op), get_vector(op)
    x  = allocate_in_domain(K); fill!(x,zero(eltype(x)))
    y  = allocate_in_domain(K); fill!(y,zero(eltype(y)))
    ns = numerical_setup(symbolic_setup(ls,K),K)
    pde_caches = (ns,K,b,x,y,uhd,assem_U)

    A,B,C = typeof(biform), typeof(liform), typeof(spaces)
    D,E,F = typeof(param),typeof(plb_caches),typeof(pde_caches)
    return new{A,B,C,D,E,F}(biform,liform,spaces,param,plb_caches,pde_caches)
  end
end

get_state(m::SelfAdjointAffineFEStateMap) = FEFunction(get_trial_space(m),m.pde_caches[4])
get_measure(m::SelfAdjointAffineFEStateMap) = m.biform.dΩ
get_spaces(m::SelfAdjointAffineFEStateMap) = m.spaces
get_assemblers(m::SelfAdjointAffineFEStateMap) = (m.pde_caches[7],m.plb_caches[2])
get_trial_space(m::SelfAdjointAffineFEStateMap) = m.spaces[1]
get_deriv_space(m::SelfAdjointAffineFEStateMap) = m.spaces[4]
get_deriv_assembler(m::SelfAdjointAffineFEStateMap) = m.plb_caches[2]

function update_forward_caches!(φ_to_u::SelfAdjointAffineFEStateMap,φh) 
  biform, liform = φ_to_u.biform, φ_to_u.liform
  U, V, _, _ = φ_to_u.spaces
  ns, K, b, x, y, uhd, assem_U = φ_to_u.pde_caches
  
  a_fwd(u,v) = biform(u,v,φh)
  l_fwd(v)   = liform(v,φh)
  assemble_matrix_and_vector!(a_fwd,l_fwd,K,b,assem_U,U,V,uhd)
  numerical_setup!(ns,K)
  return φ_to_u.pde_caches
end

function forward_solve!(φ_to_u::SelfAdjointAffineFEStateMap,φh)
  update_forward_caches!(φ_to_u,φh)
  ns, K, b, x, y, uhd, assem_U = φ_to_u.pde_caches
  solve!(x,ns,b)
  return x
end

function dRdφ(φ_to_u::SelfAdjointAffineFEStateMap,uh,vh,φh)
  biform, liform = φ_to_u.biform, φ_to_u.liform
  return ∇(biform,[uh,vh,φh],3) - ∇(liform,[vh,φh],2)
end

function update_adjoint_caches!(φ_to_u::SelfAdjointAffineFEStateMap,uh,φh)
    φ_to_u.param !== get_free_dof_values(φh) && update_forward_caches!(φ_to_u,φh)
    return φ_to_u.pde_caches 
end

function adjoint_solve!(φ_to_u::SelfAdjointAffineFEStateMap,du::AbstractVector)
    ns, K, b, x, y, uhd, assem_U = φ_to_u.pde_caches
    b !== du && solve!(y,ns,du)
    return y
end

# ConstantSelfAdjointAffineFEStateMap

struct ConstantSelfAdjointAffineFEStateMap{A,B,C,D} <: AbstractFEStateMap
  liform     :: A
  spaces     :: B
  plb_caches :: C
  pde_caches :: D

  function ConstantSelfAdjointAffineFEStateMap(
    a::Function,l::Function,
    U,V,V_φ,U_reg,φh,dΩ...;
    assem_U = SparseMatrixAssembler(U,V),
    assem_deriv = SparseMatrixAssembler(U_reg,U_reg),
    ls::LinearSolver = LUSolver()
  )
    biform = IntegrandWithMeasure(a,dΩ)
    liform = IntegrandWithMeasure(l,dΩ)
    spaces = (U,V,V_φ,U_reg)

    ## Pullback cache
    uhd = zero(U)
    vecdata = collect_cell_vector(U_reg, ∇(liform,[uhd,φh],2))
    dudφ_vec = allocate_vector(assem_deriv,vecdata)
    plb_caches = (dudφ_vec,assem_deriv)

    ## Forward cache
    op = AffineFEOperator((u,v)->biform(u,v,φh),v->liform(v,φh),U,V,assem_U)
    K, b = get_matrix(op), get_vector(op)
    x  = allocate_in_domain(K); fill!(x,zero(eltype(x)))
    y  = allocate_in_domain(K); fill!(y,zero(eltype(y)))
    ns = numerical_setup(symbolic_setup(ls,K),K)
    pde_caches = (ns,K,b,x,y,assem_U)
    A,B = typeof(liform), typeof(spaces)
    C,D = typeof(plb_caches),typeof(pde_caches)
    return new{A,B,C,D}(liform,spaces,plb_caches,pde_caches)
  end
end

get_state(m::ConstantSelfAdjointAffineFEStateMap) = FEFunction(get_trial_space(m),m.pde_caches[4])
get_measure(m::ConstantSelfAdjointAffineFEStateMap) = m.liform.dΩ
get_spaces(m::ConstantSelfAdjointAffineFEStateMap) = m.spaces
get_assemblers(m::ConstantSelfAdjointAffineFEStateMap) = (m.pde_caches[6],m.plb_caches[2])
get_trial_space(m::ConstantSelfAdjointAffineFEStateMap) = m.spaces[1]
get_deriv_space(m::ConstantSelfAdjointAffineFEStateMap) = m.spaces[4]
get_deriv_assembler(m::ConstantSelfAdjointAffineFEStateMap) = m.plb_caches[2]

function update_forward_caches!(φ_to_u::ConstantSelfAdjointAffineFEStateMap,φh) 
  liform = φ_to_u.liform
  U, V, _, _ = φ_to_u.spaces
  ns, K, b, x, y, assem_U = φ_to_u.pde_caches
  
  l_fwd(v)   = liform(v,φh)
  assemble_vector!(l_fwd,b,assem_U,V)
  return φ_to_u.pde_caches
end

function forward_solve!(φ_to_u::ConstantSelfAdjointAffineFEStateMap,φh)
  update_forward_caches!(φ_to_u,φh)
  ns, K, b, x, y, assem_U = φ_to_u.pde_caches
  solve!(x,ns,b)
  return x
end

function dRdφ(φ_to_u::ConstantSelfAdjointAffineFEStateMap,uh,vh,φh)
  liform = φ_to_u.liform
  return (-1.0)*∇(liform,[vh,φh],2)
end

function update_adjoint_caches!(φ_to_u::ConstantSelfAdjointAffineFEStateMap,uh,φh)
    return φ_to_u.pde_caches 
end

function adjoint_solve!(φ_to_u::ConstantSelfAdjointAffineFEStateMap,du::AbstractVector)
    ns, K, b, x, y, assem_U = φ_to_u.pde_caches
    b !== du && solve!(y,ns,du)
    return y
end

# Functor calls
(m::SelfAdjointAffineFEStateMap)(φh) = forward_solve!(m, φh)
(m::ConstantSelfAdjointAffineFEStateMap)(φh) = forward_solve!(m, φh)

# PDEConstrainedFunctional

struct PDEConstrainedFunctional{A}
  J
  dJ
  analytic_dJ
  state_map :: A

  function PDEConstrainedFunctional(
      objective   :: Function,
      state_map   :: AbstractFEStateMap;
      differential = nothing)
    J = StateParamIntegrandWithMeasure(objective,state_map)
    dJ = similar(J.caches[2])
    T = typeof(state_map)
    return new{T}(J,dJ,differential,state_map)
  end
end

Base.nameof(m::PDEConstrainedFunctional) = Base.nameof(m.J)
get_state(m::PDEConstrainedFunctional) = get_state(m.state_map)
get_gradient(m::PDEConstrainedFunctional) = FEFunction(get_deriv_space(m.state_map),m.dJ)

function evaluate_functionals!(pcf::PDEConstrainedFunctional,φh)
  u  = pcf.state_map(φh)
  U  = get_trial_space(pcf.state_map)
  uh = FEFunction(U,u)
  return pcf.J(uh,φh)
end

function evaluate!(pcf::PDEConstrainedFunctional,φh)
  J, dJ = pcf.J,pcf.dJ
  analytic_dJ  = pcf.analytic_dJ
  U = get_trial_space(pcf.state_map)
  U_reg = get_deriv_space(pcf.state_map)
  deriv_assem = get_deriv_assembler(pcf.state_map)
  dΩ = get_measure(pcf.state_map)

  u, u_pullback = rrule(pcf.state_map,φh)
  uh = FEFunction(U,u)

  function ∇!(F::StateParamIntegrandWithMeasure,dF,::Nothing)
    j_val, j_pullback = rrule(F,uh,φh)
    _, dFdu, dFdφ     = j_pullback(1)
    _, dφ_adj         = u_pullback(dFdu)
    copy!(dF,dφ_adj)
    dF .+= dFdφ
    return j_val
  end
  function ∇!(F::StateParamIntegrandWithMeasure,dF,dF_analytic)
    j_val = F(uh,φh)
    _dF(q) = dF_analytic(q,uh,φh,dΩ...)
    assemble_vector!(_dF,dF,deriv_assem,U_reg)
    return j_val
  end
  j = ∇!(J,dJ,analytic_dJ)
  return j,dJ
end

(pcf::PDEConstrainedFunctional)(φh) = evaluate_functionals!(pcf, φh)

# PDEsConstrainedFunctional

struct PDEsConstrainedFunctional{A,B}
  J
  dJdr
  dJdφ
  analytic_dJ
  u_state_map :: A
  r_state_map :: B

  function PDEsConstrainedFunctional(
      functional :: Function,
      u_state_map :: AbstractFEStateMap,
      r_state_map :: AbstractFEStateMap;
      differential = nothing)
    J = StateParamIntegrandWithMeasure(functional,u_state_map)
    dJdr = similar(J.caches[2])
    dJdφ = get_free_dof_values(zero(get_deriv_space(r_state_map)))
    A = typeof(u_state_map)
    B = typeof(r_state_map)
    return new{A,B}(J,dJdr,dJdφ,differential,u_state_map,r_state_map)
  end
end

Base.nameof(m::PDEsConstrainedFunctional) = Base.nameof(m.J)
get_state(m::PDEsConstrainedFunctional) = get_state(m.u_state_map)
get_gradient(m::PDEsConstrainedFunctional) = FEFunction(get_deriv_space(m.r_state_map),m.dJdφ)

function evaluate!(pcf::PDEsConstrainedFunctional,φh)
  J, dJdr, dJdφ = pcf.J,pcf.dJdr,pcf.dJdφ
  analytic_dJ  = pcf.analytic_dJ

  U = get_trial_space(pcf.u_state_map)
  R = get_trial_space(pcf.r_state_map)

  U_reg = get_deriv_space(pcf.u_state_map)
  deriv_assem = get_deriv_assembler(pcf.u_state_map)
  dΩ = get_measure(pcf.u_state_map)

  ## Foward problem
  r, r_pullback = rrule(pcf.r_state_map,φh)
  rh = FEFunction(R,r)

  u, u_pullback = rrule(pcf.u_state_map,rh)
  uh = FEFunction(U,u)

  function ∇!(F::StateParamIntegrandWithMeasure,dF,::Nothing)
    j, j_pullback = rrule(F,uh,rh)   

    _, ∂J∂u, ∂J∂r = j_pullback(1)
    _, ∂J∂u_∂u∂r  = u_pullback(∂J∂u)
    _, ∂J∂u_∂u∂φ  = r_pullback(∂J∂u_∂u∂r)

    copy!(dF, ∂J∂u_∂u∂φ)
    _, ∂J∂φ = r_pullback(∂J∂r)
    dF .+= ∂J∂φ
    return j
  end
  function ∇!(F::StateParamIntegrandWithMeasure,dF,dF_analytic)
    j_val = F(uh,rh)
    _dF(q) = dF_analytic(q,uh,rh,dΩ...)
    assemble_vector!(_dF,dJdr,deriv_assem,U_reg)
    _, dJdr_pb = r_pullback(dJdr)
    copy!(dF,dJdr_pb)
    return j_val
  end
  j = ∇!(J,dJdφ,analytic_dJ)
  return j,dJdφ
end

function evaluate!(pcf::PDEsConstrainedFunctional,φ::AbstractVector)
  V_φ = get_deriv_space(pcf.r_state_map)
  φh = FEFunction(V_φ,φ)
  return evaluate!(pcf,φh)
end

# TOPOLOGY & OPTIMIZERS

abstract type Topology end
get_state(::Topology) = @abstractmethod
get_trial_space(::Topology) = @abstractmethod

struct DensityBasedTopology <: Topology
    Ω  
    P  :: FESpace   
    ρh :: CellField 
    ρ  :: Vector{Float64}
    function DensityBasedTopology(Ω; type=:uniform, value=1.0)
        ρ_reffe = ReferenceFE(lagrangian, Float64, 0)
        Q = TestFESpace(Ω, ρ_reffe, dirichlet_tags="nodesign")
        P = TrialFESpace(Q, 1.0)
        n = num_free_dofs(P)
        if type == :uniform
            ρ = fill(value, n)
        elseif type == :random
            ρ = value*rand(n)
        end
        ρv = PVector(undef, P.gids.partition)
        get_local_values!(ρv, ρ)
        ρh = FEFunction(P, ρv)
        new(Ω,P,ρh,ρ)
    end
end
get_domain(t) = t.Ω
get_state(t::DensityBasedTopology) = t.ρh
get_trial_space(t::DensityBasedTopology) = t.P

const StateOperatorTypes = Union{Val{:ConstantSelfAdjointAffine},Val{:SelfAdjointAffine},Val{:Affine}}
const VariableTypes = Union{Val{:Scalar},Val{:Vector}}

function StateOperator(
    ::A,::B,Ω::GridapDistributed.DistributedTriangulation{D},
    liform::Function,biform::Function,
    ρ,P,dΩ...; 
    solver=:julia,
    tags="boundary",
    masks=nothing,
    vals=nothing,
    fe_order::Int=1) where D where A<:StateOperatorTypes where B<:VariableTypes

    B == Val{:Vector} ? T=VectorValue{D,Float64} : T=Float64
    reffe = ReferenceFE(lagrangian, T, fe_order)
    V = TestFESpace(Ω, reffe, dirichlet_tags=tags, dirichlet_masks=masks)
    if vals === nothing
        U = V
    else
        U = TrialFESpace(V, vals)
    end
    
    if isa(solver, Symbol)
        solver_params = default_solver_params(solver)
    elseif isa(solver, Dict)
        solver_params = solver
    end
    
    Tm = solver_params[:matrix_type]
    Tv = solver_params[:vector_type]
    assem = SparseMatrixAssembler(Tm, Tv, U, V)
    assem_deriv = SparseMatrixAssembler(Tm, Tv, P, P)
    ls = get_solver(solver_params,V)
    
    if A == Val{:ConstantSelfAdjointAffine}
        return ConstantSelfAdjointAffineFEStateMap(biform,liform,U,V,P,P,ρ,dΩ...;assem_U=assem,assem_deriv=assem_deriv,ls=ls)
    elseif A == Val{:SelfAdjointAffine}
        return SelfAdjointAffineFEStateMap(biform,liform,U,V,P,P,ρ,dΩ...;assem_U=assem,assem_deriv=assem_deriv,ls=ls)
    end
end

# Dispatch for Topology object
function StateOperator(
    a::A,b::B,
    Ω::GridapDistributed.DistributedTriangulation{D},
    liform::Function,
    biform::Function,
    t::Topology,
    dΩ...; kwargs...) where D where A<:StateOperatorTypes where B<:VariableTypes

    println("(Running Topology Dispatch for StateOperator)")
    P = get_trial_space(t)
    ρh = get_state(t)
    StateOperator(a,b,Ω,liform,biform,ρh,P,dΩ...;kwargs...)
end

# Wrapper for when Regularization (R) is present
function StateOperator(
    a::A,b::B,
    Ω::GridapDistributed.DistributedTriangulation{D},
    liform::Function,
    biform::Function,
    t::Topology,
    R::AbstractFEStateMap,
    dΩ...; kwargs...) where D where A<:StateOperatorTypes where B<:VariableTypes

    Pf = get_trial_space(R)
    ρh = get_state(t)
    _ = R(ρh)
    ρfh = get_state(R)
    StateOperator(a,b,Ω,liform,biform,ρfh,Pf,dΩ...;kwargs...)
end

function HelmholtzFilter(
    Ω,dΩ,t::Topology;
    solver=:julia,
    radius_variation=:MeshSizeProportional,
    radius_parameter=3.0)
    
    if radius_variation == :MeshSizeProportional
        h2 = get_cell_size_squared(Ω,dΩ)
        r2 = radius_parameter^2*h2
    elseif radius_variation == :Constant
        r2 = radius_parameter^2
    end
    a(u,v,ρ,dΩ) = ∫(r2 * (∇(v) ⋅ ∇(u)) + (v * u))dΩ
    l(v,ρ,dΩ) = ∫(v*ρ)dΩ

    return StateOperator(Val(:ConstantSelfAdjointAffine),Val(:Scalar),Ω,l,a,t,dΩ;tags=Int[],solver=solver)
end

function NormalizationFactor(F::Function,t::Topology,S::AbstractFEStateMap,R::AbstractFEStateMap,dΩ...;verbose=true)
    ρh = get_state(t)
    _ = R(ρh)
    ρfh = get_state(R)
    _ = S(ρfh)
    uh = get_state(S)
    F0 = sum(F(uh,ρfh,dΩ...))
    if verbose
        println("NormalizationFactor calc: ", F0)
    end
    F0 > 0.0 ? F0 = 1.0/F0 : F0=1.0
end

# Optimizers

struct history
    verbose :: Bool
    max_iter :: Int
    max_cont :: Int
    file :: String
    iter :: Vector{Int}
    freq :: Vector{Int} 
    vals :: Array{Float64,2} 
    pars :: Array{Float64} 
    names :: Vector{String}
    function history(num::Int,verbose::Bool,max_iter::Int,max_cont::Int,file_name::String,out_freq::Int,names::Vector{String})
        iter = zeros(Int,num)
        freq = fill(out_freq,num)
        vals = zeros(Float64,max_iter*max_cont,num)
        pars = zeros(Float64,max_iter*max_cont)
        new(verbose,max_iter,max_cont,file_name,iter,freq,vals,pars,names)
    end
end
verbose(h::history) = h.verbose
max_iter(h::history) = h.max_iter
write_state(h::history,field) = iszero(h.iter[field] % h.freq[field])

function serialized!(ρ::Vector{Float64},∂F∂ρ::Vector{Float64},P,F,h,field)
    Partition = P.gids.partition
    ρv = PVector(undef, Partition)
    get_local_values!(ρv, ρ)
    ρh = FEFunction(P, ρv)
    F_val, dFdρ = evaluate!(F, ρh)
    h.iter[field] += 1
    h.vals[h.iter[field],field] = F_val
    functional = nameof(F) # Get name for logging

    if verbose(h) && i_am_main(Partition) 
        println("Evaluated Functional $(functional) gives ", F_val)
    end
    if length(∂F∂ρ) > 0
        ∂F∂ρ_a = gather(own_values(dFdρ), destination=:all)
        copy_local_values!(∂F∂ρ, ∂F∂ρ_a, num_free_dofs(P))
        if write_state(h,field)
            Ω = get_triangulation(P)
            data = Vector{Any}(undef,0)
            field==1 && push!(data,"ρh"=>ρh)
            push!(data,"$(h.names[field])"=>get_state(F))
            writevtk(Ω, "$(h.file)_$(functional)_$(h.iter[field])", cellfields=data)
        end
    end
    return F_val
end

struct NLopt_MMA
    h
    opt
end

function NLopt_MMA(h::history,t::Topology,J,C::Vector;tol::Float64=1e-4)
    P = get_trial_space(t)
    opt = Opt(:LD_MMA, num_free_dofs(P))
    opt.lower_bounds = 0
    opt.upper_bounds = 1
    opt.ftol_rel = tol
    opt.maxeval = max_iter(h)
    opt.min_objective = (ρ,∂F∂ρ) -> serialized!(ρ,∂F∂ρ,P,J,h,1) 
    for (i, c) in enumerate(C)
        inequality_constraint!(opt, (ρ,∂F∂ρ) -> serialized!(ρ,∂F∂ρ,P,c,h,i+1), 1e-8)
    end
    NLopt_MMA(h,opt)
end

function optimize!(opt::NLopt_MMA,t::Topology)
    (obj_opt, ρ_opt, ret) = optimize!(opt.opt, t.ρ)
    t.ρ .= ρ_opt
    i_am_main(local_views(get_trial_space(t))) && println("Optimization finished. Return code: ", ret)
end

struct Evaluator
    h
    J
    C
end

function optimize!(opt::Evaluator,t::Topology)
    P  = get_trial_space(t)
    ρh = get_state(t)
    j_val, dJdρ = evaluate!(opt.J,ρh)
    println("Evaluator Result: ", j_val)
end

# PHYSICS

function SIMP_Elasticity(E,ν;Emin=1.0e-8,p=3)
    λ(E) = (E*ν) / ((1 + ν)*(1 - 2*ν))
    μ(E) = E / (2*(1 + ν))
    σ(ε,E) = λ(E)*tr(ε)*one(ε)  + 2*μ(E)*ε 
    ∂σ∂E(ε,E) = σ(ε,1.0)
    Eₛ(ρ) = Emin + ρ^p*(E-Emin)
    ∂Eₛ∂ρ(ρ) = -p*ρ^(p - 1)*(E-Emin)
    a(u, v, ρ, dΩ) = ∫( ε(v)⊙(σ∘(ε(u),Eₛ∘(ρ))) )dΩ
    da(u, v, ρ, δρ, dΩ) = ∫( ε(v)⊙(∂σ∂E∘(ε(u),Eₛ∘(ρ))*(∂Eₛ∂ρ∘ρ))*δρ)dΩ
    return a,da
end

# APPLICATION

function compute_navier_stokes_load(Ω, Ω_f, Ω_s; k=2, μ=0.005, ρf=0.5, U=10.0)
    Re = ρf * U / μ
    println("Reynolds number: ", Re)
    
    uf_in(x) = VectorValue( U, 0.0 ) 
    uf_0(x) = VectorValue(0.0, 0.0)
    f(x) = VectorValue(0.0, 0.0) 
    s(x) = VectorValue(0.0, 0.0) 

    reffe_u = ReferenceFE(lagrangian, VectorValue{2,Float64}, k)
    reffe_p = ReferenceFE(lagrangian, Float64, k - 1)

    Vf = TestFESpace(Ω_f, reffe_u, conformity=:H1, dirichlet_tags=["inlet", "noSlip", "interface"])
    Qf = TestFESpace(Ω_f, reffe_p, conformity=:C0)
    Y = MultiFieldFESpace([Vf, Qf])

    Uf = TrialFESpace(Vf, [uf_in, uf_0, uf_0])
    Pf = TrialFESpace(Qf)
    X = MultiFieldFESpace([Uf, Pf])

    degree = 2 * k
    dΩ_f = Measure(Ω_f, degree)
    dΩ_s = Measure(Ω_s, degree)
    Γ_fs = InterfaceTriangulation(Ω_f, Ω_s)
    n_Γfs = get_normal_vector(Γ_fs)
    dΓ_fs = Measure(Γ_fs, degree)

    σ_dev_f(ε) = 2*μ*ε 
    a_stokes((u, p), (v, q)) = ∫(ε(v) ⊙ (σ_dev_f ∘ ε(u)) - (∇ ⋅ v) * p + q * (∇ ⋅ u))dΩ_f
    conv(u, ∇u) = ρf * (∇u') ⋅ u
    dconv(du, ∇du, u, ∇u) = conv(u, ∇du) + conv(du, ∇u)
    l_f((v,q)) = ∫(v⋅f)dΩ_f

    res_f((u, p), (v, q)) = a_stokes((u, p), (v, q)) + ∫(v ⋅ (conv ∘ (u, ∇(u))))dΩ_f - l_f((v,q))
    jac_f((u, p), (du, dp), (v, q)) = a_stokes((du, dp), (v, q)) + ∫(v ⋅ (dconv ∘ (du, ∇(du), u, ∇(u))))dΩ_f

    op_f = FEOperator(res_f, jac_f, X, Y)
    nls = NewtonSolver(LUSolver(); maxiter=20, atol=1e-10, rtol=1e-7, verbose=true)
    xh = solve(nls, op_f)
    uhf, ph = xh

    l_s_vol(v,ρ,dΩ,dΓ) = ∫(v ⋅ s)dΩ
    l_s_f(v,ρ,dΩ,dΓ) = ∫( - v.⁻ ⋅ (n_Γfs.⁺ ⋅ (σ_dev_f∘ε(uhf.⁺)) - ph.⁺*n_Γfs.⁺ ) )dΓ_fs
    lₛ(v,ρ,dΩ,dΓ) = l_s_vol(v,ρ,dΩ,dΓ) + l_s_f(v,ρ,dΩ,dΓ)

    writevtk(Ω_f,"Fluid_sol",cellfields=["ph"=>ph,"uhf"=>uhf])
    return lₛ, dΩ_s, dΓ_fs
end

function NavierImmersedBodyMain(ranks, np, solvers;
    mesh_file::String,
    action=:evaluate,
    fe_order::Int=1,
    filter_radius_variation=:MeshSizeProportional,
    filter_radius_parameter::Float64=1.5,
    E::Float64=700000.0,
    ν::Float64=0.3,
    volume_fraction=0.5,
    path=".",
    title="ImmersedBody4",
    names=["uh","ρfh"],
    output_freq=1,
    verbose=false,
    max_iter::Int=3,
    continuation=:p,
    β_steps=[4.0,8.0,16.0,32.0,0,64.0],
    p_steps=[4,5,6,7,8],
    kwargs...)
    # OUTPUT DIRECTORY SETUP
    if !isdir(path)
        mkpath(path)
    end
    println("Output files will be saved to: ", abspath(path))

    solid_solver = solvers[1]
    filter_solver = solvers[2]

    model = GmshDiscreteModel(ranks, mesh_file)
    writevtk(model,"ImmersedBodyModel")

    labels = get_face_labeling(model)
    add_tag_from_tags!(labels, "design", "solid")
    add_tag_from_tags!(labels, "nodesign", "fluid")

    Ω = Interior(model)
    Ω_f = Interior(model, tags="fluid")
    Ω_s = Interior(model, tags="solid")

    lₛ, dΩ_s, dΓ_fs = compute_navier_stokes_load(Ω, Ω_f, Ω_s) 
    
    χ = DensityBasedTopology(Ω_s,type=:random,value=1.0)
    R = HelmholtzFilter(Ω_s, dΩ_s, χ; solver=filter_solver, radius_variation=filter_radius_variation, radius_parameter=filter_radius_parameter)

    p=8.0
    aₛ, daₛ = SIMP_Elasticity(E,ν;p=p)

    β = 0.01
    η = 0.5
    T(ρ) = ((tanh(β * η) + tanh(β * (ρ - η))) / (tanh(β * η) + tanh(β * (1.0 - η))))
    ∂T∂ρ(ρ) = β * (1.0 - tanh(β * (ρ - η))^2) / (tanh(β * η) + tanh(β * (1.0 - η)))

    aₚ(u, v, ρ, dΩ, dΓ) = aₛ(u, v, T ∘ ρ, dΩ)
    daₚ(u, v, ρ, δρ, dΩ, dΓ) = daₛ(u, v, T ∘ ρ, ((∂T∂ρ ∘ ρ) * δρ), dΩ)

    S = StateOperator(Val(:SelfAdjointAffine), Val(:Vector), Ω_s, lₛ, aₚ, χ, R, dΩ_s, dΓ_fs; tags="fixed", solver=solid_solver)

    J1(u, ρ, dΩ, dΓ) = aₚ(u, u, ρ, dΩ, dΓ)
    J0 = NormalizationFactor(J1, χ, S, R, dΩ_s, dΓ_fs)
    J(u, ρ, dΩ, dΓ) = J0 * J1(u, ρ, dΩ, dΓ)
    dJ(δρ, u, ρ, dΩ, dΓ) = J0 * daₚ(u, u, ρ, δρ, dΩ, dΓ)
    Jc = PDEsConstrainedFunctional(J, S, R; differential=dJ)

    V0 = sum(∫(1)dΩ_s)
    V(ϱ, ρ, dΩ) = (1.0 / V0) * ∫((T ∘ ϱ) - volume_fraction)dΩ
    dV(δρ, ϱ, ρ, dΩ) = (1.0 / V0) * ∫(∂T∂ρ ∘ (ϱ) * δρ)dΩ
    Vc = PDEConstrainedFunctional(V, R; differential=dV)

    if continuation == :β
        max_cont = length(β_steps) + 1
    elseif continuation == :p   
        max_cont = length(p_steps) + 1
    else
        max_cont = 1
    end

    h = history(2,verbose,max_iter,max_cont,joinpath(path,title),output_freq,names)
    h.pars .= β
    if action == :evaluate
        opt = Evaluator(h, Jc, []; kwargs...)
    elseif action == :optimize_w_mma
        opt = NLopt_MMA(h, χ, Jc, [Vc]; kwargs...)
    end

    optimize!(opt, χ)
    if continuation == :β
        for β_i in β_steps
            β = β_i 
            opt.h.pars[h.iter[1]+1:end].=β
            optimize!(opt, χ)
        end
    elseif continuation == :p
        for p_i in p_steps
            p = p_i 
            opt.h.pars[h.iter[1]+1:end].=p
            optimize!(opt, χ)
        end
    end

    io = open("history.txt","w")
    write(io,"$h")
    close(io)
    return nothing 
end

function NavierImmersedBody(;
    solid_solver=:julia,
    filter_solver=:julia, kwargs...)
    solvers = [solid_solver, filter_solver]
    run_with_backend(NavierImmersedBodyMain, solvers; kwargs...)
end

# EXECUTION

NavierImmersedBody(
    backend=:sequential,
    np=(1,1),
    mesh_file=mesh_file_path,
    action=:optimize_w_mma,
    solid_solver=:julia,
    filter_solver=:julia,
    volume_fraction=0.5,
    max_iter=100,
    verbose=true,
    path="./results",
    title="navierimmersed_optimized",
)
