const _clblast_status_codes = Dict{Int, Symbol}(
    # Status codes in common with the OpenCL standard
        0=>:CLBlastSuccess                   , # CL_SUCCESS
       -3=>:CLBlastOpenCLCompilerNotAvailable, # CL_COMPILER_NOT_AVAILABLE
       -4=>:CLBlastTempBufferAllocFailure    , # CL_MEM_OBJECT_ALLOCATION_FAILURE
       -5=>:CLBlastOpenCLOutOfResources      , # CL_OUT_OF_RESOURCES
       -6=>:CLBlastOpenCLOutOfHostMemory     , # CL_OUT_OF_HOST_MEMORY
      -11=>:CLBlastOpenCLBuildProgramFailure , # CL_BUILD_PROGRAM_FAILURE: OpenCL compilation error
      -30=>:CLBlastInvalidValue              , # CL_INVALID_VALUE
      -36=>:CLBlastInvalidCommandQueue       , # CL_INVALID_COMMAND_QUEUE
      -38=>:CLBlastInvalidMemObject          , # CL_INVALID_MEM_OBJECT
      -42=>:CLBlastInvalidBinary             , # CL_INVALID_BINARY
      -43=>:CLBlastInvalidBuildOptions       , # CL_INVALID_BUILD_OPTIONS
      -44=>:CLBlastInvalidProgram            , # CL_INVALID_PROGRAM
      -45=>:CLBlastInvalidProgramExecutable  , # CL_INVALID_PROGRAM_EXECUTABLE
      -46=>:CLBlastInvalidKernelName         , # CL_INVALID_KERNEL_NAME
      -47=>:CLBlastInvalidKernelDefinition   , # CL_INVALID_KERNEL_DEFINITION
      -48=>:CLBlastInvalidKernel             , # CL_INVALID_KERNEL
      -49=>:CLBlastInvalidArgIndex           , # CL_INVALID_ARG_INDEX
      -50=>:CLBlastInvalidArgValue           , # CL_INVALID_ARG_VALUE
      -51=>:CLBlastInvalidArgSize            , # CL_INVALID_ARG_SIZE
      -52=>:CLBlastInvalidKernelArgs         , # CL_INVALID_KERNEL_ARGS
      -53=>:CLBlastInvalidLocalNumDimensions , # CL_INVALID_WORK_DIMENSION: Too many thread dimensions
      -54=>:CLBlastInvalidLocalThreadsTotal  , # CL_INVALID_WORK_GROUP_SIZE: Too many threads in total
      -55=>:CLBlastInvalidLocalThreadsDim    , # CL_INVALID_WORK_ITEM_SIZE: ... or for a specific dimension
      -56=>:CLBlastInvalidGlobalOffset       , # CL_INVALID_GLOBAL_OFFSET
      -57=>:CLBlastInvalidEventWaitList      , # CL_INVALID_EVENT_WAIT_LIST
      -58=>:CLBlastInvalidEvent              , # CL_INVALID_EVENT
      -59=>:CLBlastInvalidOperation          , # CL_INVALID_OPERATION
      -61=>:CLBlastInvalidBufferSize         , # CL_INVALID_BUFFER_SIZE
      -63=>:CLBlastInvalidGlobalWorkSize     , # CL_INVALID_GLOBAL_WORK_SIZE
    #
    # Status codes in common with the clBLAS library
    -1024=>:CLBlastNotImplemented            , # Routine or functionality not implemented yet
    -1022=>:CLBlastInvalidMatrixA            , # Matrix A is not a valid OpenCL buffer
    -1021=>:CLBlastInvalidMatrixB            , # Matrix B is not a valid OpenCL buffer
    -1020=>:CLBlastInvalidMatrixC            , # Matrix C is not a valid OpenCL buffer
    -1019=>:CLBlastInvalidVectorX            , # Vector X is not a valid OpenCL buffer
    -1018=>:CLBlastInvalidVectorY            , # Vector Y is not a valid OpenCL buffer
    -1017=>:CLBlastInvalidDimension          , # Dimensions M, N, and K have to be larger than zero
    -1016=>:CLBlastInvalidLeadDimA           , # LD of A is smaller than the matrix's first dimension
    -1015=>:CLBlastInvalidLeadDimB           , # LD of B is smaller than the matrix's first dimension
    -1014=>:CLBlastInvalidLeadDimC           , # LD of C is smaller than the matrix's first dimension
    -1013=>:CLBlastInvalidIncrementX         , # Increment of vector X cannot be zero
    -1012=>:CLBlastInvalidIncrementY         , # Increment of vector Y cannot be zero
    -1011=>:CLBlastInsufficientMemoryA       , # Matrix A's OpenCL buffer is too small
    -1010=>:CLBlastInsufficientMemoryB       , # Matrix B's OpenCL buffer is too small
    -1009=>:CLBlastInsufficientMemoryC       , # Matrix C's OpenCL buffer is too small
    -1008=>:CLBlastInsufficientMemoryX       , # Vector X's OpenCL buffer is too small
    -1007=>:CLBlastInsufficientMemoryY       , # Vector Y's OpenCL buffer is too small
    #
    # Custom additional status codes for CLBlast
    -2050=>:CLBlastInsufficientMemoryTemp    , # Temporary buffer provided to GEMM routine is too small
    -2049=>:CLBlastInvalidBatchCount         , # The batch count needs to be positive
    -2048=>:CLBlastInvalidOverrideKernel     , # Trying to override parameters for an invalid kernel
    -2047=>:CLBlastMissingOverrideParameter  , # Missing override parameter(s) for the target kernel
    -2046=>:CLBlastInvalidLocalMemUsage      , # Not enough local memory available on this device
    -2045=>:CLBlastNoHalfPrecision           , # Half precision (16-bits) not supported by the device
    -2044=>:CLBlastNoDoublePrecision         , # Double precision (64-bits) not supported by the device
    -2043=>:CLBlastInvalidVectorScalar       , # The unit-sized vector is not a valid OpenCL buffer
    -2042=>:CLBlastInsufficientMemoryScalar  , # The unit-sized vector's OpenCL buffer is too small
    -2041=>:CLBlastDatabaseError             , # Entry for the device was not found in the database
    -2040=>:CLBlastUnknownError              , # A catch-all error code representing an unspecified error
    -2039=>:CLBlastUnexpectedError              # A catch-all error code representing an unexpected exception
)

struct CLBlastError <: Exception
    code::Int
    desc::Symbol

    function CLBlastError(c::Integer)
        new(c, get(_clblast_status_codes, Int(c), :CL_UNKNOWN_ERROR_CODE))
    end
end

Base.show(io::IO, err::CLBlastError) = print(io, "CLBlastError(code=$(err.code), $(err.desc))")
