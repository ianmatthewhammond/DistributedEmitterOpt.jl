"""
    RunLogConfig

Configuration for runtime logging during optimization/solver execution.

All logging is opt-in (`enabled=false` by default). If enabled, logs can be
routed to stdout, file, or both, with per-category toggles and level filtering.

Environment overrides:
- `DEO_LOG_ENABLE=1` enables logger by default
- `DEO_LOG_LEVEL=DEBUG|INFO|WARN|ERROR` sets minimum level
- `DEO_LOG_FORCE_STDOUT=1` forces flush after each event
"""

using Dates
using Logging

function _env_bool(name::AbstractString, default::Bool=false)
    raw = get(ENV, name, "")
    isempty(raw) && return default
    v = lowercase(strip(raw))
    v in ("1", "true", "yes", "on")
end

function _env_loglevel(default::LogLevel=Logging.Info)
    raw = uppercase(strip(get(ENV, "DEO_LOG_LEVEL", "")))
    isempty(raw) && return default
    if raw == "DEBUG"
        return Logging.Debug
    elseif raw == "INFO"
        return Logging.Info
    elseif raw == "WARN"
        return Logging.Warn
    elseif raw == "ERROR"
        return Logging.Error
    end
    return default
end

Base.@kwdef struct RunLogConfig
    enabled::Bool = _env_bool("DEO_LOG_ENABLE", false)
    min_level::LogLevel = _env_loglevel(Logging.Info)
    sink::Symbol = :stdout          # :stdout | :file | :both
    path::Union{Nothing,String} = nothing
    force_flush::Bool = _env_bool("DEO_LOG_FORCE_STDOUT", false)
    every_iter::Int = 1

    # Category toggles
    epoch::Bool = true
    objective::Bool = true
    memory::Bool = false
    solver::Bool = false
    constraints::Bool = true
end

mutable struct RunLogger
    cfg::RunLogConfig
    fileio::Union{Nothing,IO}
end

const _RUN_LOGGER = Ref{Union{Nothing,RunLogger}}(nothing)

active_run_logger() = _RUN_LOGGER[]

function _resolve_log_path(cfg::RunLogConfig, root::Union{Nothing,String})
    if isnothing(cfg.path)
        base = isnothing(root) ? pwd() : root
        return joinpath(base, "run.log")
    end
    isabspath(cfg.path) && return cfg.path
    isnothing(root) ? cfg.path : joinpath(root, cfg.path)
end

function _open_log_file(cfg::RunLogConfig, root::Union{Nothing,String})
    cfg.sink in (:file, :both) || return nothing
    path = _resolve_log_path(cfg, root)
    mkpath(dirname(path))
    open(path, "a")
end

function _close_run_logger!(logger::RunLogger)
    if !isnothing(logger.fileio)
        try
            close(logger.fileio)
        catch
        end
    end
    return nothing
end

function activate_run_logger!(cfg::RunLogConfig; root::Union{Nothing,String}=nothing)
    deactivate_run_logger!()
    cfg.enabled || return nothing
    _RUN_LOGGER[] = RunLogger(cfg, _open_log_file(cfg, root))
    return _RUN_LOGGER[]
end

function deactivate_run_logger!()
    logger = active_run_logger()
    if !isnothing(logger)
        _close_run_logger!(logger)
    end
    _RUN_LOGGER[] = nothing
    return nothing
end

function with_run_logger(f::Function, cfg::RunLogConfig; root::Union{Nothing,String}=nothing)
    prev = active_run_logger()
    logger = nothing
    if cfg.enabled
        logger = RunLogger(cfg, _open_log_file(cfg, root))
        _RUN_LOGGER[] = logger
    end
    try
        return f()
    finally
        if !isnothing(logger)
            _close_run_logger!(logger)
        end
        _RUN_LOGGER[] = prev
    end
end

@inline _level_value(level::LogLevel) = Int(getfield(level, :level))

function _category_enabled(cfg::RunLogConfig, category::Symbol)
    if category === :epoch
        return cfg.epoch
    elseif category === :objective
        return cfg.objective
    elseif category === :memory
        return cfg.memory
    elseif category === :solver
        return cfg.solver
    elseif category === :constraints
        return cfg.constraints
    else
        return true
    end
end

@inline function _iter_enabled(cfg::RunLogConfig, iter::Union{Nothing,Int})
    isnothing(iter) && return true
    cfg.every_iter <= 1 && return true
    return iter % cfg.every_iter == 0
end

function _format_kv(kwargs)
    isempty(kwargs) && return ""
    parts = String[]
    for (k, v) in kwargs
        push!(parts, string(k, "=", repr(v)))
    end
    return join(parts, " ")
end

function _emit_stdout(level::LogLevel, line::String)
    lv = _level_value(level)
    if lv <= _level_value(Logging.Debug)
        @debug line
    elseif lv <= _level_value(Logging.Info)
        @info line
    elseif lv <= _level_value(Logging.Warn)
        @warn line
    else
        @error line
    end
end

function _maybe_flush(cfg::RunLogConfig, fileio::Union{Nothing,IO}=nothing)
    cfg.force_flush || return nothing
    flush(stdout)
    Libc.flush_cstdio()
    if !isnothing(fileio)
        flush(fileio)
    end
    return nothing
end

function _write_file(fileio::IO, level::LogLevel, category::Symbol, line::String)
    tstamp = Dates.format(Dates.now(), dateformat"yyyy-mm-ddTHH:MM:SS")
    println(fileio, "[$tstamp][$(_level_value(level))][$category] $line")
    return nothing
end

"""
    log_event(level, category, message; iter=nothing, kwargs...)

Emit a structured runtime log event. When a run logger is active, category and
level filters are enforced. Without an active run logger, info/warn/error events
still pass through standard Julia logging for backwards compatibility.
"""
function log_event(level::LogLevel, category::Symbol, message::AbstractString;
    iter::Union{Nothing,Int}=nothing, kwargs...)
    logger = active_run_logger()
    kv = _format_kv(kwargs)
    line = isempty(kv) ? "[$category] $message" : "[$category] $message $kv"
    if !isnothing(iter)
        line = "[$category] iter=$iter $message" * (isempty(kv) ? "" : " $kv")
    end

    if isnothing(logger)
        if _level_value(level) >= _level_value(Logging.Info)
            _emit_stdout(level, line)
        end
        return nothing
    end

    cfg = logger.cfg
    if _level_value(level) < _level_value(cfg.min_level)
        return nothing
    end
    if !_category_enabled(cfg, category) || !_iter_enabled(cfg, iter)
        return nothing
    end

    if cfg.sink in (:stdout, :both)
        _emit_stdout(level, line)
    end
    if cfg.sink in (:file, :both) && !isnothing(logger.fileio)
        _write_file(logger.fileio, level, category, line)
    end
    _maybe_flush(cfg, logger.fileio)
    return nothing
end

log_debug(category::Symbol, message::AbstractString; kwargs...) =
    log_event(Logging.Debug, category, message; kwargs...)

log_info(category::Symbol, message::AbstractString; kwargs...) =
    log_event(Logging.Info, category, message; kwargs...)

log_warn(category::Symbol, message::AbstractString; kwargs...) =
    log_event(Logging.Warn, category, message; kwargs...)

log_error(category::Symbol, message::AbstractString; kwargs...) =
    log_event(Logging.Error, category, message; kwargs...)
