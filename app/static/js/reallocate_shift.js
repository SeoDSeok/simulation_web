document.addEventListener('DOMContentLoaded', () => {
    const doctorBase = [34, 30];
    const nurseBase = [3, 7, 3];

    function generateOptions(base) {
        const delta = Math.max(1, Math.floor(base * 0.1));
        const options = [];
        for (let i = base - delta; i <= base + delta; i++) {
            options.push(`<option value="${i}">${i}</option>`);
        }
        return options.join('');
    }

    function createShiftFields(role, base, keepTotal, containerId) {
        const container = document.getElementById(containerId);
        container.innerHTML = '';
        const roleLower = role.toLowerCase();
        const total = base.reduce((a, b) => a + b, 0);

        for (let i = 0; i < base.length; i++) {
            const label = `${role} Shift ${i + 1}:`;
            const name = `${roleLower}_shift_${i}`;
            if (keepTotal && i === base.length - 1) {
                container.innerHTML += `
                    <label>${label}</label>
                    <input type="number" name="${name}" id="${name}" readonly>
                `;
            } else {
                container.innerHTML += `
                    <label>${label}</label>
                    <select name="${name}" id="${name}">
                        ${generateOptions(base[i])}
                    </select>
                `;
            }
        }
    }

    function recalculate(role, base) {
        const keepTotal = document.getElementById(`keep_total_${role.toLowerCase()}`).value === 'yes';
        if (!keepTotal) return;

        const roleLower = role.toLowerCase();
        const total = base.reduce((a, b) => a + b, 0);
        let sum = 0;
        for (let i = 0; i < base.length - 1; i++) {
            const val = parseInt(document.getElementById(`${roleLower}_shift_${i}`).value || 0);
            sum += val;
        }
        const finalInput = document.getElementById(`${roleLower}_shift_${base.length - 1}`);
        if (finalInput) finalInput.value = Math.max(0, total - sum);
    }

    function setup(role, base, containerId) {
        const keepTotalEl = document.getElementById(`keep_total_${role.toLowerCase()}`);
        keepTotalEl.addEventListener('change', () => {
            createShiftFields(role, base, keepTotalEl.value === 'yes', containerId);
            setTimeout(() => recalculate(role, base), 100);
        });

        document.addEventListener('change', (e) => {
            if (e.target.name && e.target.name.startsWith(`${role.toLowerCase()}_shift_`)) {
                recalculate(role, base);
            }
        });

        // 초기화
        createShiftFields(role, base, keepTotalEl.value === 'yes', containerId);
        setTimeout(() => recalculate(role, base), 100);
    }

    setup('Doctor', doctorBase, 'doctor_shift_inputs');
    setup('Nurse', nurseBase, 'nurse_shift_inputs');
});
