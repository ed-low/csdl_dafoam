#!/bin/bash
# ===============================================
# CFD Optimization Log → Roadmap Compiler Script
# Preserves original line numbers from console_output.txt
# ===============================================
# Usage: ./compile_roadmap_with_lines.sh console_output.txt
# ===============================================

# ---------------------------
# Check input
# ---------------------------
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 console_output.txt"
    exit 1
fi

RAW_FILE="$1"

# ---------------------------
# Stage 0: Prepare event patterns
# ---------------------------
PATTERNS_FILE="event_patterns.txt"
cat > "$PATTERNS_FILE" << EOL
Running Primal Solver
Primal solution converged!
Primal solution failed!
Computing d[drag]/d[dafoam_solver_states]
Solving Linear Equation...
PetscConvergedReason:
Computing d[aero_residuals]/d[aero_vol_coords]^T
dRdWTPC:
Major iteration:
Inside SLSQP line search
alpha successful without nan
Stepping back x_k_new
Merit function or its gradient at the new point has NaN
Objective function is NaN or Inf
Objective gradient contains NaN or Inf
Returning NANs
EOL
#Adjoint solution failed!
#EOL

echo "Stage 0: Event patterns saved to $PATTERNS_FILE"

# ---------------------------
# Stage 1: Strip setup + primal timestep noise
# Preserve original line numbers
# ---------------------------
STAGE1="stage_1_stripped.txt"

# Step 1a: prefix lines with original numbers
awk '{print NR ":" $0}' "$RAW_FILE" > numbered_console.txt

# Step 1b: remove everything before last "evaluating geometry component elapsed time"
awk -F: '
/evaluating geometry component elapsed time/ {last=NR}
{lines[NR]=$0}
END {
    for(i=last+1;i<=NR;i++) print lines[i]
}
' numbered_console.txt > stage_1_tmp.txt

# Step 1c: remove primal timestep noise blocks
awk -F: '
/Starting time loop/ {skip=1}
skip && /End/ {skip=0; next}
!skip
' stage_1_tmp.txt > "$STAGE1"

echo "Stage 1: Setup and primal timestep noise removed → $STAGE1"

# ---------------------------
# Stage 2: Extract only key event lines
# ---------------------------
STAGE2="stage_2_events.txt"

grep -F -f "$PATTERNS_FILE" "$STAGE1" > "$STAGE2"

echo "Stage 2: Key event lines extracted → $STAGE2"

# ---------------------------
# Stage 3: Deduplicate repeated MPI echoes
# ---------------------------
STAGE3="stage_3_deduped.txt"

awk -F: '
{
    line_num = $1
    msg = substr($0, index($0,$2))
    if(msg != prev) {
        print $0
        prev = msg
    }
}
' "$STAGE2" > "$STAGE3"

echo "Stage 3: MPI duplicates removed → $STAGE3"

# ---------------------------
# Stage 4: Semantic rewrite → readable roadmap
# ---------------------------
ROADMAP="optimization_roadmap.txt"

awk -F: '
BEGIN {
    printf("%-8s | %-18s | %s\n", "LINE", "EVENT", "DETAILS")
    printf("------------------------------------------------------------\n")
    last_event = ""
    in_backtrack = 0        # true when we are inside a backtracking episode
    last_failure_msg = ""   # remembers last failure text to allow new failures to show
}

{
    line = $1
    msg  = substr($0, index($0,$2))
}

# helper to emit an event unless it is filtered by backtracking rules
function emit(event, details,    out) {
    out = details
    # Always emit if event changes (keeps timeline readable)
    if (event != last_event) {
        printf("%-8s | %-18s | %s\n", line, event, out)
        last_event = event
    } else {
        # If the same event repeats but not in a suppressed backtrack context,
        # still emit (some events are legitimately repeated).
        # For backtracking-managed events we skip here; caller controls in_backtrack.
        # For other events, emit once (already done) and skip duplicates.
        # (No-op)
    }
}

# reset backtracking suppression (call when a meaningful state change happens)
function reset_backtrack() {
    in_backtrack = 0
    last_failure_msg = ""
    last_event = ""
}

# ---- reset triggers ----
/Major iteration:/ {
    reset_backtrack()
    emit("MAJOR ITERATION", msg)
    next
}

/Running Primal Solver/ {
    reset_backtrack()
    sub(/^.*Running Primal Solver /,"",msg)
    primal_id = msg
    next
}

/Primal solution converged!/ {
    reset_backtrack()
    emit("PRIMAL SOLVE", "solver " primal_id " CONVERGED")
    next
}

/Primal solution failed!/ {
    reset_backtrack()
    emit("PRIMAL SOLVE", "solver " primal_id " FAILED")
    next
}

# If alpha accepted, that ends the backtracking episode
/alpha successful without nan/ {
    emit("LINE SEARCH", "alpha accepted")
    reset_backtrack()
    next
}

# ---- adjoint solve (keeps semantics) ----
/Solving Linear Equation/ {
    adj_line = line
    adj_pending = 1
    next
}

/PetscConvergedReason:/ && adj_pending {
    if (msg ~ /PetscConvergedReason: 2/) {
        emit("ADJOINT SOLVE", "CONVERGED")
    } else {
        emit("ADJOINT SOLVE", "FAILED")
    }
    adj_pending = 0
    # adjoint result is a meaningful change — reset backtracking
    reset_backtrack()
    next
}

# ---- gradients / jacobians / preconditioner ----
/Computing d\[drag\]\/d\[dafoam_solver_states\]/ {
    emit("GRADIENT", "dF/dW")
    next
}

/Computing d\[aero_residuals\]\/d\[aero_vol_coords\]\^T/ {
    emit("GRADIENT", "dR/dX")
    next
}

/dRdWTPC:/ {
    emit("PRECONDITIONER", "dR/dW TPC")
    next
}

# ---- backtracking / NaN-style failures ----
# |Returning NANs|Adjoint solution failed! Returning NANs|NaN|Inf|NANs/
# Treat failure messages specially so we show the first failure in a backtracking episode,
# then suppress repeated identical failures until a reset or a different failure string appears.
/Merit function or its gradient at the new point has NaN or inf.|Objective gradient contains NaN|Objective function is NaN or Inf./ {
    # if this failure message is new OR we are not currently suppressing backtracking, emit it
    if (!in_backtrack || msg != last_failure_msg) {
        emit("FAILURE", msg)
        last_failure_msg = msg
        in_backtrack = 1
    } else {
        # suppressed duplicate failure — do nothing
    }
    next
}

# ---- line-search backtracking messages ----
/Stepping back x_k_new|Inside SLSQP line search|Reperforming backtracking line search|backtracking/ {
    # emit the first LINE SEARCH with "backtracking" per episode
    if (!in_backtrack) {
        emit("LINE SEARCH", "backtracking")
        in_backtrack = 1
    } else {
        # suppressed duplicate LINE SEARCH during same backtracking episode
    }
    next
}

# ---- alpha success (already handled above) ----
# ---- any other failure-terminating phrases could be added here ----

# ---- generic fallback: emit other interesting events if not identical ----
{
    # For any other messages we still want to keep (rare), just emit once and reset nothing.
    # (This avoids spamming.)
    # You can add more patterns above if you want finer control.
    next
}

' "$STAGE3" > "$ROADMAP"

echo "Stage 4: Semantic roadmap generated → $ROADMAP"
echo "All stages complete."

