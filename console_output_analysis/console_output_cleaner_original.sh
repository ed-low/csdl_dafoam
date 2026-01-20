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
Adjoint solution failed!
EOL

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
}

{
    line = $1
    msg  = substr($0, index($0,$2))
}

# Primal solves
msg ~ /Running Primal Solver/ {
    sub(/^.*Running Primal Solver /,"",msg)
    primal_id = msg
    next
}

msg ~ /Primal solution converged!/ {
    printf("%-8s | %-18s | solver %s CONVERGED\n", line, "PRIMAL SOLVE", primal_id)
    next
}

msg ~ /Primal solution failed!/ {
    printf("%-8s | %-18s | solver %s FAILED\n", line, "PRIMAL SOLVE", primal_id)
    next
}

msg ~ /Computing d\[drag\]\/d\[dafoam_solver_states\]/ {
    printf("%-8s | %-18s | dF/dW\n", line, "GRADIENT")
    next
}

msg ~ /Computing d\[aero_residuals\]\/d\[aero_vol_coords\]\^T/ {
    printf("%-8s | %-18s | dR/dX\n", line, "JACOBIAN")
    next
}

msg ~ /dRdWTPC:/ {
    printf("%-8s | %-18s | dR/dW TPC\n", line, "PRECONDITIONER")
    next
}

msg ~ /Solving Linear Equation/ {
    adj_line = line
    adj_pending = 1
    next
}

msg ~ /PetscConvergedReason:/ && adj_pending {
    if(msg ~ /PetscConvergedReason: 2/) {
        printf("%-8s | %-18s | CONVERGED\n", adj_line, "ADJOINT SOLVE")
    } else {
        printf("%-8s | %-18s | FAILED\n", adj_line, "ADJOINT SOLVE")
    }
    adj_pending = 0
    next
}

msg ~ /Major iteration:/ {
    printf("%-8s | %-18s | %s\n", line, "MAJOR ITERATION", msg)
    next
}

msg ~ /alpha successful without nan/ {
    printf("%-8s | %-18s | alpha accepted\n", line, "LINE SEARCH")
    next
}

msg ~ /Stepping back x_k_new/ {
    printf("%-8s | %-18s | backtracking\n", line, "LINE SEARCH")
    next
}

msg ~ /NaN|Inf|NANs/ {
    printf("%-8s | %-18s | %s\n", line, "FAILURE", msg)
    next
}
' "$STAGE3" > "$ROADMAP"

echo "Stage 4: Semantic roadmap generated → $ROADMAP"
echo "All stages complete."

