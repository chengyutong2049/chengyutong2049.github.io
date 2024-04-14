---
layout: default
title: HW 2
parent: System and Software Security
grand_parent: VT Courses
permalink: /docs/vt-courses/software-security/hw-2
---

CS/ECE 5590 System and Software Security (Spring 2024)
{: .fs-9 }
<!-- ./grace/main.py is the main file to kick off experiments. -->
{: .fs-6 .fw-300 }

# Written Homework Assignment 2

## 1. Know your system and hardware security acronyms.

| Security Acronyms | Full Name | Security Acronyms | Full Name | 
|:----------------- |:-------------------------------------- |:----------------- |:----------------------------------------- | 
| TEE | Trusted Execution Environment | MAC | Media Access Control | 
| TPM | Trusted Platform Module | DAC | Discretionary Access Control | 
| SGX | Software Guard Extensions | BLP model | Bell-LaPadula model | 
| TCB | Trusted Computing Base | CSRF | Cross-Site Request Forgery |

## 2. OS integrity verification.

### a.	What are the 2 properties of cryptographic hash functions?
1. **One-way**: It is computationally infeasible to find any input that maps to any pre-specified output. 
2. **Collision resistant**: It is computationally infeasible to find any two distinct inputs that map to the same output.

### b.	Describe step-by-step how TripWire solution utilizes cryptographic hashes to preserve system integrity.
1. Signature Model Selection:

- Tripwire supports different signature routines, including MD5, Snefru, SHA, and others, allowing for up to ten signatures to be used for each file. This allows system administrators to tailor the balance between performance and security according to local policy.

1. Signature Generation (siggen):
- Tripwire includes a program called `siggen`, which generates signatures for the files specified on the command line. This tool supports the generation of any of the signature types included in the Tripwire distribution for any file.

3. Building Tripwire:
- The system administrator starts by loading a clean distribution of the operating system onto an isolated machine. Following the unpacking of the Tripwire distribution, system-specific tools are specified, and a `conf-machine.h` header file is chosen or created to describe special options for the machine to be monitored.
- After configuration, `make` is used to build the Tripwire binaries. `make test` starts the Tripwire test suite, which exercises all the signature routines to ensure correct signature generation, and compares all the Tripwire source files against a test database to ensure distribution integrity.

4. Installing the Database:
- The system administrator then builds the system database using the `tripwire -initialize` command in single-user mode to ensure no tampering. The database builds based on the `tw.config` file, which contains a listing of all directories and files to be scanned, with their associated selection-masks. After initialization, the database is reminded to be moved to read-only media.

5. Integrity Checking:
- To check file system integrity, Tripwire compares the current state of the file system against the database. It scans the file system to determine whether any files have been added, deleted, or changed. Running `tripwire` generates a report of these files.
- Alternatively, `tripwire -interactive` runs Tripwire in interactive mode, where for each added, deleted, or altered file found, the user is prompted whether or not the database entry should be updated. This ensures that the database can be updated securely without overwriting the existing data.

### c.	Under what threat model may the TripWire fail (i.e., the TripWire verification is successful, even when kernel files are compromised)?
1. **Rootkits and Advanced Persistent Threats (APTs)**: If an attacker gains root access to a system, they could potentially install a rootkit that hides changes made to the kernel files. Advanced persistent threats often use sophisticated methods to stay undetected for long periods. A rootkit could intercept TripWire’s attempts to verify file integrities, returning falsified, unaltered data to TripWire even when changes have been made.

2. **Time of Check to Time of Use (TOCTOU) Race Conditions**: If there's a time gap between when TripWire checks a file and when it uses the information gained from that check, an attacker could exploit this window to modify the files after they've been checked but before TripWire acts on the results. This could mean TripWire might not detect alterations made in this very narrow time window.

3. **Direct Memory Access (DMA) Attacks**: If an attacker has physical access to a machine, they could potentially use DMA-capable devices (e.g., through FireWire, Thunderbolt ports) to modify the kernel memory directly, bypassing the file system and, therefore, TripWire’s monitoring.

4. **Compromised TripWire Binaries**: If the TripWire binaries themselves are compromised or replaced by a malicious version (perhaps through a prior breach or supply chain attack), the integrity checks could inaccurately report that the kernel files are unmodified, even when they are not. This assumes the attacker has the capability to suppress any alerts about the TripWire modification itself.

5. **Steganographic Information Hiding**: Though more sophisticated and less likely, an attacker could manipulate the files in such a way that alterations are hidden within data that appears unmodified to TripWire. By using steganographic techniques, the modifications might not change the file attributes (like size, timestamp) that TripWire checks.

6. **Sophisticated Encryption or Obfuscation Techniques**: Attackers could utilize sophisticated encryption or obfuscation techniques to modify kernel files in a way that TripWire's integrity check algorithms cannot decipher the changes. Such techniques can hide the alteration of the content effectively from TripWire's standard verification processes.

7. **Filesystem Manipulation or Corruption**: In some advanced scenarios, attackers might manipulate the filesystem metadata or induce corruption in a way that misleads TripWire about the actual contents of the files it intends to check, thus evading detection

### d. Briefly explain how the above security issue can be solved using TPM.
1. **Rootkits and APTs**: TPM can ensure that the boot process is secure by verifying the integrity of the system and its components before loading the OS, significantly reducing the chances for rootkits and APTs to take hold. By leveraging TPM-based attestation, the integrity of the TripWire binaries and other critical components can be verified, ensuring they haven't been tampered with.

2. **TOCTOU Race Conditions**: By using TPM, cryptographic hashes of critical files and configurations can be stored securely and used to detect any changes or modifications. Since TPM is a separate, isolated module, it's much harder for an attacker to manipulate these hashes, effectively addressing TOCTOU issues.

3. **DMA Attacks**: TPM helps mitigate DMA attacks by ensuring secure boot and providing a trusted execution environment. It can ensure that only trusted, integrity-verified code runs on the system, helping prevent attackers from running arbitrary code or modifying the system via DMA.

4. **Compromised TripWire Binaries**: TPM's secure boot process can verify the integrity of system and application binaries, including TripWire, before they are executed. This helps ensure that compromised binaries are detected before they can do harm.

5. **Steganographic Information Hiding**: The comprehensive integrity checks supported by TPM, including hash measurements of software components, can help detect even subtle changes that might be hidden through steganography, as long as those changes alter the content that contributes to the integrity measurement.

6. **Sophisticated Encryption or Obfuscation Techniques**: TPM's secure keystore can be used to manage encryption keys more securely, making it harder for attackers to apply encryption or obfuscation in a way that's undetectable. TPM also enhances the overall encryption infrastructure, making it more resistant to attacks.

7. **Filesystem Manipulation or Corruption**: TPM can store secure hashes of filesystem metadata and critical files, helping to ensure that any manipulation or corruption of the filesystem can be detected more reliably.

## 3. Answer the following questions (according to [Youtube Video](https://www.youtube.com/watch?v=h0wSJyYGGY4), [Paper](https://ethz.ch/content/dam/ethz/special-interest/infk/inst-infsec/system-security-group-dam/research/publications/pub2020/CODASPY2020.pdf)).
### a.	What’s the main security capability provided by Intel SGX?
Intel SGX (Software Guard Extensions) provides a main security capability known as **remote attestation**. This feature is crucial as it allows a remote verifier to ensure that a software application is running in a secure manner within an Intel SGX enclave on an untrusted platform before sensitive data is provisioned to it. This helps in maintaining confidentiality and integrity of the application's data even if the platform it runs on is compromised.

### b. Please briefly describe step-by-step the procedure of the standard/typical SGX remote attestation between an enclave and a remote verifier.
1. **Establishing Trust**: The remote verifier needs to ensure that the target computing platform is running the expected secure enclave. This verification is crucial before provisioning any secrets to the enclave.
2. **Attestation Request**: The remote verifier sends an attestation request to the target platform, where the enclave is located.
3. **Local Attestation by Enclave**: The enclave on the target platform then generates an attestation response. This response includes evidence that the enclave is running as expected, typically by leveraging Intel’s enhanced privacy ID (EPID) group signatures.
4. **Relay Vulnerability Consideration**: In standard attestation, the protocol does not ensure that the attestation response comes directly from the targeted computing platform itself rather than a relayed source. This means an adversarial OS, or other software on the platform, could potentially redirect these requests to another platform under their control (although the document discusses new methods in ProximiTEE to counter such vulnerabilities).
5. **Remote Verification**: Upon receiving the attestation response, the remote verifier uses Intel SGX's attestation keys to verify the EPID group signatures. This verifies that the correct enclave generated the signatures.
6. **Provisioning Secrets**: If the attestation response is verified successfully, the remote verifier proceeds to provision secrets to the enclave, which might include sensitive data, cryptographic keys, or other secrets needed by the application running within the enclave.

### c. The authors identified a security issue in the above standard remote attestation procedure. What is it?
The security issue identified by the authors in the standard remote attestation procedure of Intel SGX is its vulnerability to relay attacks. The standard attestation mechanism doesn't guarantee that the enclave runs on the expected computing platform. An adversary with control over the OS or other software on the target platform can relay incoming attestation requests to another platform. This could grant the adversary increased capabilities such as mounting physical side-channel attacks—including both physical and digital attacks—that would not have been possible without the relay. This issue significantly undermines the security assumption of SGX attestation as it allows the attestation to be redirected and verified even when the enclave is not on the intended or secure platform.

### d. Use 1-2 sentences to explain the authors’ proposed solution to that security problem.
The authors propose "ProximiTEE," a novel solution for preventing SGX remote attestation relay attacks. This solution employs a trusted embedded device attached to the target computing platform. The device verifies the proximity of the attested enclave to ensure attestation integrity, overcoming malicious software such as a compromised OS on the platform. This setup allows for secure and periodic attestation, maintaining the security even in complex deployment scenarios.