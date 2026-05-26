def check_ec8a_logic(numbers):

    result = {
        "status": "VALID",
        "issues": []
    }

    try:
        # Expected order:
        # [registered, accredited, issued, unused,
        # spoiled, rejected, valid, used]

        registered = numbers[0]
        accredited = numbers[1]
        issued = numbers[2]
        unused = numbers[3]
        spoiled = numbers[4]
        rejected = numbers[5]
        valid = numbers[6]
        used = numbers[7]

        # RULE 1:
        # Used ballots = valid + rejected + spoiled
        if used != (valid + rejected + spoiled):
            result["status"] = "TAMPERED"
            result["issues"].append(
                "Used ballots do not equal valid + rejected + spoiled"
            )

        # RULE 2:
        # Issued ballot papers = unused + valid + rejected + spoiled
        if issued != (unused + valid + rejected + spoiled):
            result["status"] = "TAMPERED"
            result["issues"].append(
                "Issued ballot papers do not equal unused + valid + rejected + spoiled"
            )

        # RULE 3:
        # Accredited voters must not exceed registered voters
        if accredited > registered:
            result["status"] = "TAMPERED"
            result["issues"].append(
                "Accredited voters exceed registered voters"
            )

        # RULE 4:
        # Valid votes must not exceed accredited voters
        if valid > accredited:
            result["status"] = "TAMPERED"
            result["issues"].append(
                "Valid votes exceed accredited voters"
            )

    except Exception:
        result["status"] = "REVIEW"
        result["issues"].append(
            "Could not fully extract EC8A numbers"
        )

    return result